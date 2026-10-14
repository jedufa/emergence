#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Nightly gate:
# - full strong regression including strict replay + longrun stage
# - additional multi-seed evolvability longrun sweep for stronger drift detection
tests/regression_strong.sh --fail-fast "$@"
tests/evolvability/evolvability_longrun_checkpoint_resume.sh --seeds 42,1337 --resume-gens 10,20 --steps-per-generation 8

