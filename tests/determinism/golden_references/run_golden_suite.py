#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load_catalog(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError("catalog must be JSON object")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description="Run tiered golden replay suite profiles")
    ap.add_argument(
        "profile",
        nargs="?",
        default="quick",
        choices=["quick", "standard", "nightly"],
        help="suite profile (default: quick)",
    )
    ap.add_argument(
        "--profile",
        dest="profile_flag",
        choices=["quick", "standard", "nightly"],
        help="backward-compatible profile flag (overrides positional)",
    )
    ap.add_argument("--catalog", default=str(Path(__file__).resolve().parent / "golden_suite_catalog.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    profile = str(args.profile_flag or args.profile)

    catalog = _load_catalog(Path(args.catalog))
    profiles = catalog.get("profiles", {})
    cases = catalog.get("cases", [])
    if not isinstance(profiles, dict) or not isinstance(cases, list):
        raise RuntimeError("invalid catalog schema")

    id_to_case = {}
    for c in cases:
        if isinstance(c, dict) and isinstance(c.get("id"), str):
            id_to_case[c["id"]] = c

    ids = profiles.get(profile, [])
    if not isinstance(ids, list):
        raise RuntimeError(f"profile '{profile}' is invalid")

    nodeids: list[str] = []
    print(f"[golden-suite] profile={profile} cases={len(ids)}")
    for cid in ids:
        c = id_to_case.get(str(cid))
        if c is None:
            raise RuntimeError(f"missing case id: {cid}")
        nodeid = str(c.get("nodeid", "")).strip()
        if not nodeid:
            raise RuntimeError(f"case without nodeid: {cid}")
        print(f"- {cid}: {c.get('topic','')} -> {nodeid}")
        nodeids.append(nodeid)

    cmd = ["pytest", "-q", *nodeids]
    print("[golden-suite] cmd:", " ".join(cmd))
    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
