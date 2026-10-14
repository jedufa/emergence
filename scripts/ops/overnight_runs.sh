#!/usr/bin/env bash
# Compatibility wrapper for legacy script path.
set -euo pipefail
exec "$(dirname "$0")/../../tools/legacy/scripts/ops/overnight_runs.sh" "$@"
