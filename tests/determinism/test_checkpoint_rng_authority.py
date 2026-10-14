from __future__ import annotations

from pathlib import Path
import re

def test_checkpoint_rng_v2_authoritative_save_semantics_present():
    # This is a structural guard: ensure checkpoint code indicates v2 authority to prevent dual-source drift.
    repo_root = Path(__file__).resolve().parents[2]
    p = repo_root / "engine" / "simulation" / "checkpoint.py"
    assert p.exists(), f"Expected file missing: {p}"
    txt = p.read_text(encoding="utf-8", errors="replace")

    # We expect code to mention rng_snapshot_v2.json and to *suppress* legacy RNG payload when v2 is present.
    # These regexes are intentionally loose: they verify intent without pinning exact line numbers.
    assert "rng_snapshot_v2" in txt, "checkpoint.py does not mention rng_snapshot_v2"
    # Look for wording indicating v2 preferred/authoritative OR that legacy payload is suppressed/emptied when v2 exists.
    has_prefer = re.search(r"prefer\s+.*rng_snapshot_v2", txt, re.IGNORECASE) is not None
    has_suppress = re.search(r"(suppress|empty|omit).*legacy.*rng", txt, re.IGNORECASE) is not None
    assert has_prefer or has_suppress, "checkpoint.py lacks an explicit v2-authority / legacy-suppression signal"
