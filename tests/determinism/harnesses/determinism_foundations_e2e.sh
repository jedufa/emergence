#!/usr/bin/env bash
set -euo pipefail

# Phase-0 Determinism E2E Harness (P0-A..P0-E + phase3 resume determinism)
# Non-interactive, fail-fast, CI-safe.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

UTC_NOW="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_BASE_DEFAULT="${REPO_ROOT}/runs/phase0_determinism_e2e"
OUT_DIR="${OUT_BASE_DEFAULT}/phase0_e2e_${UTC_NOW}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --out-dir requires a path" >&2; exit 2; }
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: tests/determinism/harnesses/determinism_foundations_e2e.sh [--out-dir <path>]

Runs unattended determinism gates for P0-A..P0-E plus phase3 resume determinism.
Writes logs and summaries under runs/phase0_determinism_e2e by default.
EOF
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

LOG_DIR="${OUT_DIR}/logs"
STAGE_JSONL="${OUT_DIR}/stage_results.jsonl"
SUMMARY_TXT="${OUT_DIR}/phase0_e2e_summary.txt"
SUMMARY_JSON="${OUT_DIR}/phase0_e2e_summary.json"
PHASE3_RUNS_DIR="${REPO_ROOT}/runs/phase3_resume_determinism_tmp"

mkdir -p "${LOG_DIR}"
: > "${STAGE_JSONL}"

export PYTHONHASHSEED=0
export EIEL_DETERMINISM_STRICT=1
export EIEL_FORENSIC_V4=1
export EIEL_POLICY_EXEC_PROBE=1
export CI=1

OVERALL_STATUS="PASS"
FAILED_STAGE=""

log_line() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

append_stage_json() {
  local stage_name="$1"
  local stage_status="$2"
  local elapsed_s="$3"
  local log_path="$4"
  local cmd_text="$5"
  local note_text="${6:-}"
  python - "$STAGE_JSONL" "$stage_name" "$stage_status" "$elapsed_s" "$log_path" "$cmd_text" "$note_text" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
obj = {
    "stage": sys.argv[2],
    "status": sys.argv[3],
    "elapsed_seconds": float(sys.argv[4]),
    "log_path": sys.argv[5],
    "command": sys.argv[6],
    "note": sys.argv[7],
}
with path.open("a", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")
PY
}

finalize_summary() {
  python - "$STAGE_JSONL" "$SUMMARY_TXT" "$SUMMARY_JSON" "$OUT_DIR" "$OVERALL_STATUS" "$FAILED_STAGE" <<'PY'
import json
import os
import sys
from pathlib import Path

stage_jsonl = Path(sys.argv[1])
summary_txt = Path(sys.argv[2])
summary_json = Path(sys.argv[3])
out_dir = Path(sys.argv[4])
overall_status = sys.argv[5]
failed_stage = sys.argv[6]

summary_txt.parent.mkdir(parents=True, exist_ok=True)
summary_json.parent.mkdir(parents=True, exist_ok=True)

stages = []
if stage_jsonl.exists():
    for raw in stage_jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                stages.append(obj)
        except Exception:
            continue

payload = {
    "schema_version": "phase0_determinism_e2e_summary_v1",
    "overall_status": overall_status,
    "failed_stage": failed_stage or None,
    "out_dir": str(out_dir),
    "env": {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
        "EIEL_DETERMINISM_STRICT": os.environ.get("EIEL_DETERMINISM_STRICT", ""),
        "EIEL_FORENSIC_V4": os.environ.get("EIEL_FORENSIC_V4", ""),
        "EIEL_POLICY_EXEC_PROBE": os.environ.get("EIEL_POLICY_EXEC_PROBE", ""),
    },
    "stages": stages,
}

summary_json.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

lines = [
    f"overall_status: {overall_status}",
    f"failed_stage: {failed_stage or '-'}",
    f"out_dir: {out_dir}",
    "",
    "stages:",
]
for s in stages:
    lines.append(
        f"- {s.get('stage')} | status={s.get('status')} | elapsed={s.get('elapsed_seconds')}s | log={s.get('log_path')}"
    )
    if s.get("note"):
        lines.append(f"  note: {s.get('note')}")
summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

trap 'finalize_summary' EXIT

run_stage() {
  local stage_name="$1"
  local cmd_text="$2"
  local log_file="${LOG_DIR}/${stage_name}.log"
  local start_s
  local elapsed_s
  local rc
  local note=""

  start_s="${SECONDS}"
  log_line "STAGE START: ${stage_name}"
  log_line "COMMAND: ${cmd_text}"
  set +e
  bash -lc "${cmd_text}" > "${log_file}" 2>&1
  rc=$?
  set -e
  elapsed_s="$((SECONDS - start_s))"

  if [[ "${stage_name}" == "p0c_torch_migration" ]]; then
    if grep -q "CUDA unavailable" "${log_file}"; then
      note="CUDA unavailable skip observed (non-fatal on CPU-only hosts)"
    fi
  fi

  if [[ "${rc}" -eq 0 ]]; then
    append_stage_json "${stage_name}" "PASS" "${elapsed_s}" "${log_file}" "${cmd_text}" "${note}"
    log_line "STAGE PASS: ${stage_name} (${elapsed_s}s)"
  else
    append_stage_json "${stage_name}" "FAIL" "${elapsed_s}" "${log_file}" "${cmd_text}" "${note}"
    OVERALL_STATUS="FAIL"
    FAILED_STAGE="${stage_name}"
    log_line "STAGE FAIL: ${stage_name} (${elapsed_s}s) log=${log_file}"
    return 1
  fi
}

log_line "Output directory: ${OUT_DIR}"
log_line "Repository root: ${REPO_ROOT}"

# P0-A: scanner + lock + gate
run_stage "p0a_static_rng_scan_json" "python -m tools.diagnostics.static_rng_scan --repo_root . --json"
run_stage "p0a_gate_with_tests" "tools/diagnostics/verify_p0a_determinism_gate.sh --with-tests"

# P0-B: RNG authority/injection invariants via existing tests
run_stage "p0b_rng_authority_runtime_tests" "pytest -q -ra tests/determinism/test_checkpoint_rng_authority.py tests/determinism/test_no_rng_islands_runtime.py"

# P0-C/P0-D/P0-E focused tests
run_stage "p0c_torch_migration" "pytest -q -ra tests/determinism/p0/test_p0c_torch_rng_migration.py"
run_stage "p0d_numpy_python_migration" "pytest -q -ra tests/determinism/p0/test_p0d_numpy_python_rng_migration.py"
run_stage "p0e_rng_budget_enforcement" "pytest -q -ra tests/determinism/p0/test_p0e_rng_budget_enforcement.py"

# Required checkpoint/resume anchor command (hard gate)
run_stage "phase3_resume_determinism_e2e" \
  "PYTHONHASHSEED=0 EIEL_FORENSIC_V4=1 EIEL_POLICY_EXEC_PROBE=1 EIEL_DETERMINISM_STRICT=1 CLEAN_RUNS=1 ZIP_OUT_DIR=runs RUNS_DIR=${PHASE3_RUNS_DIR} ./tests/determinism/harnesses/checkpoint_resume_determinism_e2e.sh --membership-trace --membership-keep 64 --membership-harness --resume-gen 10 --resume-gens 10"

log_line "ALL STAGES PASS"
exit 0
