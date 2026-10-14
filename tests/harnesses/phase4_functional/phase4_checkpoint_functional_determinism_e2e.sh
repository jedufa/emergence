#!/usr/bin/env bash
# phase4_checkpoint_functional_determinism_e2e.sh
#
# Thin CI-friendly wrapper for the Phase 4 functional determinism harness.
# The heavy lifting lives in the Python module under tests/.

set -euo pipefail

exec python -m tests.harnesses.phase4_functional.phase4_functional_determinism_harness "$@"
