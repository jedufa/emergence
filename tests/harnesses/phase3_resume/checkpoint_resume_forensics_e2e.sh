#!/usr/bin/env bash
# Last modified: 2026-03-05
#
# Dedicated replay-forensics profile:
# - keeps strict replay equivalence as primary gate
# - enables state-dump defaults so membership harness can run deep diff
#
# Usage:
#   PYTHONHASHSEED=0 EIEL_FORENSIC_V4=1 EIEL_POLICY_EXEC_PROBE=1 EIEL_STRICT_DETERMINISM=1 \
#   CLEAN_RUNS=1 ZIP_OUT_DIR=runs \
#   ./tests/determinism/harnesses/checkpoint_resume_forensics_e2e.sh \
#     --membership-trace --membership-keep 64 --membership-harness --resume-gen 10 --resume-gens 10 \
#     -- --knob sim_cfg.steps_per_generation=8

set -euo pipefail

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export EIEL_FORENSIC_V4="${EIEL_FORENSIC_V4:-1}"
export EIEL_POLICY_EXEC_PROBE="${EIEL_POLICY_EXEC_PROBE:-1}"
export EIEL_STRICT_DETERMINISM="${EIEL_STRICT_DETERMINISM:-1}"

exec ./tests/determinism/harnesses/checkpoint_resume_determinism_e2e.sh \
  --forensic-membership-profile \
  "$@"

