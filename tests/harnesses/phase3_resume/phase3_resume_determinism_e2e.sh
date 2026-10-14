#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[DEPRECATED] Use tests/harnesses/phase3_resume/checkpoint_resume_determinism_e2e.sh" >&2
exec bash "$SCRIPT_DIR/checkpoint_resume_determinism_e2e.sh" "$@"
