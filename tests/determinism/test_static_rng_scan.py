from __future__ import annotations

"""P0-A: static determinism enforcement smoke test.

This test is intentionally simple:
- Run the repo-wide static scanner on the runtime surface.
- Fail if any non-allowlisted forbidden RNG usage exists.
- Fail if the allowlist lock is missing or out of date.
"""

from pathlib import Path


def test_p0a_static_rng_scan_passes():
    from tools.diagnostics.static_rng_scan import scan_repo

    repo_root = Path(__file__).resolve().parents[2]
    hits, summary = scan_repo(repo_root)

    assert summary.get("lock_ok", False), f"Allowlist lock failed: {summary.get('lock_reason','')}"
    violations = [h for h in hits if not h.allowlisted]
    assert not violations, f"Forbidden RNG usage detected: count={len(violations)} (see tools.diagnostics.static_rng_scan output)"
