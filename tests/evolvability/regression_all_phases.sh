#!/usr/bin/env bash
set -euo pipefail

# Evolvability regression gate (DET-FND -> Phase E).
# Runs the curated command matrix in deterministic CI-safe mode.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

UTC_NOW="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR_DEFAULT="${REPO_ROOT}/runtime/logs/validation/evolvability/regression_all_phases_${UTC_NOW}"
OUT_DIR="${OUT_DIR_DEFAULT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --out-dir requires a path" >&2; exit 2; }
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: tests/evolvability/regression_all_phases.sh [--out-dir <path>]

Runs full curated regression sequence across determinism and evolvability phases.
Writes per-stage logs and summary artifacts under runtime/logs/validation/evolvability by default.
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
SUMMARY_JSON="${OUT_DIR}/regression_summary.json"
SUMMARY_TXT="${OUT_DIR}/regression_summary.txt"
COMMANDS_TXT="${OUT_DIR}/commands.txt"

mkdir -p "${LOG_DIR}"
: > "${STAGE_JSONL}"
: > "${COMMANDS_TXT}"

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
  python - "$STAGE_JSONL" "$stage_name" "$stage_status" "$elapsed_s" "$log_path" "$cmd_text" <<'PY'
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
    "schema_version": "eiel_evolvability_regression_summary_v1",
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
summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

trap 'finalize_summary' EXIT

run_stage() {
  local stage_name="$1"
  local cmd_text="$2"
  local log_file="${LOG_DIR}/${stage_name}.log"
  local start_s elapsed_s rc

  start_s="${SECONDS}"
  log_line "STAGE START: ${stage_name}"
  log_line "COMMAND: ${cmd_text}"
  printf '%s\n' "${cmd_text}" >> "${COMMANDS_TXT}"

  set +e
  bash -lc "${cmd_text}" > "${log_file}" 2>&1
  rc=$?
  set -e
  elapsed_s="$((SECONDS - start_s))"

  if [[ "${rc}" -eq 0 ]]; then
    append_stage_json "${stage_name}" "PASS" "${elapsed_s}" "${log_file}" "${cmd_text}"
    log_line "STAGE PASS: ${stage_name} (${elapsed_s}s)"
  else
    append_stage_json "${stage_name}" "FAIL" "${elapsed_s}" "${log_file}" "${cmd_text}"
    OVERALL_STATUS="FAIL"
    FAILED_STAGE="${stage_name}"
    log_line "STAGE FAIL: ${stage_name} (${elapsed_s}s) log=${log_file}"
    return 1
  fi
}

log_line "Output directory: ${OUT_DIR}"
log_line "Repository root: ${REPO_ROOT}"

run_stage "01_static_rng_scan" "python -m tools.diagnostics.static_rng_scan --repo_root . --json"
run_stage "02_verify_p0a" "tools/diagnostics/verify_p0a_determinism_gate.sh --with-tests"
run_stage "03_phase1_registry_smoke" "python experiments/runs/checkpointed_autopilot_influence_phase1_registry_smoke.py"
run_stage "04_pytest_p0" "pytest -q tests/determinism/p0"
run_stage "05_pytest_phase_a" "pytest -q tests/evolvability/phase_a"
run_stage "06_pytest_phase_b" "pytest -q tests/evolvability/phase_b"
run_stage "07_pytest_phase_c" "pytest -q tests/evolvability/phase_c"
run_stage "08_pytest_phase_d" "pytest -q tests/evolvability/phase_d"
run_stage "09_pytest_phase_e" "pytest -q tests/evolvability/phase_e"
run_stage "10_pytest_epi_engine" "pytest -q tests/integration/metrics/test_epi_engine.py"
run_stage "11_pytest_phasee_plus_regressions" "pytest -q tests/evolvability/phase_e tests/evolvability/phase_d/test_d1_developmental_program.py tests/evolvability/phase_c/test_phase_c_integration_validation.py tests/evolvability/phase_b/test_b1_modular_genome_canonicalization.py tests/integration/metrics/test_epi_engine.py"
run_stage "12_pytest_full" "pytest -q"

log_line "ALL STAGES PASS"
exit 0
