"""Run the representative source-release CPU gate, with exclusions made explicit."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
# Kept intact in source, independently characterized in VALIDATION.md. The
# first pins serialized bytes from a different historical toolchain; the second
# is an old complete-tree architecture baseline, not this curated source scope.
HISTORICAL_SENTINELS = (
    "tests/modular_contracts/test_crca_checkpoint_occurrence_v1.py::test_occurrence_wrapper_preserves_exact_bytes_and_loader_semantics",
    "tests/modular_contracts/test_modular_import_guard_v1.py::test_live_e922_baseline_is_self_consistent_and_green",
)
TARGETS = (
    "tests/modular_contracts", "tests/packaging", "tests/release", "tests/security",
    "tests/test_eiel_runtime_roots.py", "tests/test_eiel_evidence_root_portability.py",
    "tests/test_eiel_evidence_manager_v1.py", "tests/test_eiel_evidence_manager_determinism_v11.py",
    "tests/test_eiel_evidence_manager_codex_guards_v1_1.py",
    "tests/test_eiel_root_retirement_v1.py", "tests/test_eiel_campaign_analysis_v1.py",
)


def main() -> int:
    command = [sys.executable, "-m", "pytest", "-q", "-ra", "-o", "addopts=", *TARGETS]
    command += [f"--deselect={node}" for node in HISTORICAL_SENTINELS]
    print("Representative CPU gate; historical sentinel exclusions:", flush=True)
    for node in HISTORICAL_SENTINELS:
        print("  " + node, flush=True)
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
