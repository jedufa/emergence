#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Fast PR-oriented gate:
# - full strong regression stack
# - skip heavy replay stages (21/22)
tests/regression_strong.sh --fail-fast --skip-replay "$@"

