#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[DEPRECATED] Use tests/determinism/harnesses/determinism_foundations_e2e.sh" >&2
exec bash "$SCRIPT_DIR/determinism_foundations_e2e.sh" "$@"
