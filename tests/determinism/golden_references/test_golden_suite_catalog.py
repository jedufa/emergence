from __future__ import annotations

import json
from pathlib import Path


CATALOG = Path(__file__).resolve().parent / "golden_suite_catalog.json"


def _load() -> dict:
    with open(CATALOG, "r", encoding="utf-8") as f:
        obj = json.load(f)
    assert isinstance(obj, dict)
    return obj


def _nodeid_exists(nodeid: str) -> bool:
    if "::" not in nodeid:
        return False
    p, fn = nodeid.split("::", 1)
    fp = Path(p)
    if not fp.exists() or not fp.is_file():
        return False
    txt = fp.read_text(encoding="utf-8", errors="replace")
    return f"def {fn}(" in txt


def test_golden_suite_catalog_counts_and_profiles() -> None:
    c = _load()
    cases = c.get("cases", [])
    assert isinstance(cases, list)
    assert len(cases) == 24

    tiers = c.get("tiers", {})
    assert tiers == {"mini": 12, "macro": 8, "mega": 4}

    by_tier = {"mini": 0, "macro": 0, "mega": 0}
    for case in cases:
        assert isinstance(case, dict)
        t = str(case.get("tier", ""))
        assert t in by_tier
        by_tier[t] += 1
    assert by_tier == {"mini": 12, "macro": 8, "mega": 4}

    profiles = c.get("profiles", {})
    assert isinstance(profiles, dict)
    quick = profiles.get("quick", [])
    standard = profiles.get("standard", [])
    nightly = profiles.get("nightly", [])
    assert isinstance(quick, list) and len(quick) == 6
    assert isinstance(standard, list) and len(standard) == 16
    assert isinstance(nightly, list) and len(nightly) == 24


def test_golden_suite_catalog_nodeids_exist() -> None:
    c = _load()
    cases = c.get("cases", [])
    for case in cases:
        assert isinstance(case, dict)
        nid = str(case.get("nodeid", ""))
        assert nid
        assert _nodeid_exists(nid), f"missing nodeid target: {nid}"
