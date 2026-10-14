#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
exec "${REPO_ROOT}/tests/harnesses/phase3_resume/checkpoint_resume_forensics_e2e.sh" "$@"
