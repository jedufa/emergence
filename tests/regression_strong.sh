#!/usr/bin/env bash
set -euo pipefail

# Strong deterministic regression gate for multi-agent development.
# Goals:
# 1) Single fail-fast orchestrator for critical deterministic checks.
# 2) Emit immutable regression_manifest.json with command + outcome evidence.
# 3) Hash critical replay/golden artifacts into runs/.../regression_hashes.json.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

UTC_NOW="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR_DEFAULT="${REPO_ROOT}/runtime/logs/validation/evolvability/regression_strong_${UTC_NOW}"
OUT_DIR="${OUT_DIR_DEFAULT}"
RUNS_DIR=""
SKIP_REPLAY=0
DRY_RUN=0
FAIL_FAST=0
START_AT=1
TESTS_FILTER_RAW=""
TESTS_FILTER_ACTIVE=0
declare -A TESTS_FILTER_NUMS=()
declare -A TESTS_FILTER_MATCHED=()

usage() {
cat <<'EOF'
Usage: tests/regression_strong.sh [options]

Options:
  --out-dir <path>      Output root (default: runtime/logs/validation/evolvability/regression_strong_<utc>)
  --runs-dir <path>     Replay output root (default: <out-dir>/runs)
  --start_at <n>        Start from stage number n (default: 1)
  --start-at <n>        Alias of --start_at
  --tests <csv>         Run only selected stage numbers (e.g. 5,11,17)
  --skip-replay         Skip strict replay harness stage
  --fail-fast           Stop at first failed stage (default: run all stages)
  --dry-run             Print stage commands and exit without executing
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --out-dir requires a path" >&2; exit 2; }
      OUT_DIR="$2"
      shift 2
      ;;
    --runs-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --runs-dir requires a path" >&2; exit 2; }
      RUNS_DIR="$2"
      shift 2
      ;;
    --start_at|--start-at)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires an integer stage number" >&2; exit 2; }
      START_AT="$2"
      shift 2
      ;;
    --tests)
      [[ $# -ge 2 ]] || { echo "ERROR: --tests requires csv stage numbers" >&2; exit 2; }
      TESTS_FILTER_RAW="$2"
      TESTS_FILTER_ACTIVE=1
      IFS=',' read -r -a __tests_tokens <<< "${TESTS_FILTER_RAW}"
      for __tok in "${__tests_tokens[@]}"; do
        __trim="${__tok//[[:space:]]/}"
        [[ -n "${__trim}" ]] || continue
        if ! [[ "${__trim}" =~ ^[0-9]+$ ]]; then
          echo "ERROR: --tests entries must be integers, got: ${__trim}" >&2
          exit 2
        fi
        __n=$((10#${__trim}))
        TESTS_FILTER_NUMS["${__n}"]=1
      done
      if [[ "${#TESTS_FILTER_NUMS[@]}" -eq 0 ]]; then
        echo "ERROR: --tests produced empty selection" >&2
        exit 2
      fi
      shift 2
      ;;
    --skip-replay)
      SKIP_REPLAY=1
      shift
      ;;
    --fail-fast)
      FAIL_FAST=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! [[ "${START_AT}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --start_at must be an integer, got: ${START_AT}" >&2
  exit 2
fi
if (( START_AT < 1 || START_AT > 99 )); then
  echo "ERROR: --start_at out of allowed range [1,99]: ${START_AT}" >&2
  exit 2
fi

if [[ -z "${RUNS_DIR}" ]]; then
  RUNS_DIR="${OUT_DIR}/runs"
fi

LOG_DIR="${OUT_DIR}/logs"
STAGE_JSONL="${OUT_DIR}/stage_results.jsonl"
SUMMARY_JSON="${OUT_DIR}/regression_summary.json"
MANIFEST_JSON="${OUT_DIR}/regression_manifest.json"
COMMANDS_TXT="${OUT_DIR}/commands.txt"
HASHES_JSON="${RUNS_DIR}/regression_hashes.json"

mkdir -p "${LOG_DIR}" "${RUNS_DIR}"
: > "${STAGE_JSONL}"
: > "${COMMANDS_TXT}"

# Determinism-oriented defaults for this gate.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export EIEL_FORENSIC_V4="${EIEL_FORENSIC_V4:-1}"
export EIEL_POLICY_EXEC_PROBE="${EIEL_POLICY_EXEC_PROBE:-1}"
export EIEL_DETERMINISM_STRICT="${EIEL_DETERMINISM_STRICT:-1}"
export EIEL_STRICT_DETERMINISM="${EIEL_STRICT_DETERMINISM:-1}"
export DEBUG_DETERMINISM="${DEBUG_DETERMINISM:-1}"
export CI="${CI:-1}"

OVERALL_STATUS="PASS"
FAILED_STAGE=""

log_line() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

print_stage_result() {
  local status="$1"
  local stage="$2"
  # Respect NO_COLOR and non-tty outputs.
  if [[ -n "${NO_COLOR:-}" || ! -t 1 ]]; then
    echo "${status}: ${stage}"
    return 0
  fi
  local green red reset
  green=$'\033[32m'
  red=$'\033[31m'
  reset=$'\033[0m'
  if [[ "${status}" == "PASS" ]]; then
    printf "%sPASS%s: %s\n" "${green}" "${reset}" "${stage}"
  else
    printf "%sFAIL%s: %s\n" "${red}" "${reset}" "${stage}"
  fi
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

write_summary_and_manifest() {
  python - "$STAGE_JSONL" "$SUMMARY_JSON" "$MANIFEST_JSON" "$OUT_DIR" "$RUNS_DIR" "$OVERALL_STATUS" "$FAILED_STAGE" "$COMMANDS_TXT" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

stage_jsonl = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
manifest_json = Path(sys.argv[3])
out_dir = Path(sys.argv[4])
runs_dir = Path(sys.argv[5])
overall_status = sys.argv[6]
failed_stage = sys.argv[7] or None
commands_txt = Path(sys.argv[8])

stages = []
if stage_jsonl.exists():
    for raw in stage_jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            stages.append(obj)

summary_payload = {
    "schema_version": "eiel_regression_strong_summary_v1",
    "overall_status": overall_status,
    "failed_stage": failed_stage,
    "out_dir": str(out_dir),
    "runs_dir": str(runs_dir),
    "stages": stages,
}
summary_json.write_text(json.dumps(summary_payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

critical_files = []

# Golden run reports (stage logs) + orchestrator evidence.
for rel in [
    "logs/03_golden_quick.log",
    "logs/04_golden_standard.log",
    "logs/05_golden_nightly.log",
    "stage_results.jsonl",
    "regression_summary.json",
]:
    p = out_dir / rel
    if p.exists() and p.is_file():
        critical_files.append(p)

# Replay outputs and core run artifacts.
if runs_dir.exists():
    for p in sorted(runs_dir.rglob("*")):
        if not p.is_file():
            continue
        name = p.name
        if (
            name == "metrics.csv"
            or name == "run_summary.json"
            or "equivalence" in name
            or "determinism_fingerprint" in name
            or "membership" in name
        ):
            critical_files.append(p)

hash_items = []
seen = set()
for p in critical_files:
    rp = str(p.resolve())
    if rp in seen:
        continue
    seen.add(rp)
    hash_items.append(
        {
            "path": str(p),
            "sha256": _sha256(p),
            "size_bytes": int(p.stat().st_size),
        }
    )

hashes_payload = {
    "schema_version": "eiel_regression_hashes_v1",
    "overall_status": overall_status,
    "items": hash_items,
}
runs_dir.mkdir(parents=True, exist_ok=True)
(runs_dir / "regression_hashes.json").write_text(
    json.dumps(hashes_payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

git_commit = ""
try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    git_commit = ""

commands = []
if commands_txt.exists():
    commands = [ln.strip() for ln in commands_txt.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]

manifest_payload = {
    "schema_version": "eiel_regression_manifest_v1",
    "git_commit": git_commit,
    "overall_status": overall_status,
    "failed_stage": failed_stage,
    "paths": {
        "out_dir": str(out_dir),
        "runs_dir": str(runs_dir),
        "summary_json": str(summary_json),
        "hashes_json": str(runs_dir / "regression_hashes.json"),
        "stage_results_jsonl": str(stage_jsonl),
    },
    "env": {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
        "EIEL_FORENSIC_V4": os.environ.get("EIEL_FORENSIC_V4", ""),
        "EIEL_POLICY_EXEC_PROBE": os.environ.get("EIEL_POLICY_EXEC_PROBE", ""),
        "EIEL_DETERMINISM_STRICT": os.environ.get("EIEL_DETERMINISM_STRICT", ""),
        "EIEL_STRICT_DETERMINISM": os.environ.get("EIEL_STRICT_DETERMINISM", ""),
        "DEBUG_DETERMINISM": os.environ.get("DEBUG_DETERMINISM", ""),
        "CI": os.environ.get("CI", ""),
    },
    "commands": commands,
    "stages": stages,
}
manifest_json.write_text(json.dumps(manifest_payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

trap 'write_summary_and_manifest' EXIT

run_stage() {
  local stage_name="$1"
  local cmd_text="$2"
  local log_file="${LOG_DIR}/${stage_name}.log"
  local stage_prefix stage_num
  local start_s elapsed_s rc

  stage_prefix="${stage_name%%_*}"
  if [[ "${stage_prefix}" =~ ^([0-9]+) ]]; then
    stage_num=$((10#${BASH_REMATCH[1]}))
  else
    echo "ERROR: stage name does not start with numeric prefix: ${stage_name}" >&2
    return 2
  fi

  if (( stage_num < START_AT )); then
    append_stage_json "${stage_name}" "SKIP_START_AT" "0" "${log_file}" "${cmd_text}"
    log_line "STAGE SKIP: ${stage_name} (start_at=${START_AT})"
    return 0
  fi
  if [[ "${TESTS_FILTER_ACTIVE}" == "1" && -z "${TESTS_FILTER_NUMS[${stage_num}]+x}" ]]; then
    append_stage_json "${stage_name}" "SKIP_FILTER" "0" "${log_file}" "${cmd_text}"
    log_line "STAGE SKIP: ${stage_name} (tests filter)"
    return 0
  fi
  if [[ "${TESTS_FILTER_ACTIVE}" == "1" ]]; then
    TESTS_FILTER_MATCHED["${stage_num}"]=1
  fi

  log_line "STAGE START: ${stage_name}"
  log_line "COMMAND: ${cmd_text}"
  printf '%s\n' "${cmd_text}" >> "${COMMANDS_TXT}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    append_stage_json "${stage_name}" "SKIP_DRY_RUN" "0" "${log_file}" "${cmd_text}"
    return 0
  fi

  start_s="${SECONDS}"
  set +e
  bash -lc "${cmd_text}" > "${log_file}" 2>&1
  rc=$?
  set -e
  elapsed_s="$((SECONDS - start_s))"

  if [[ "${rc}" -eq 0 ]]; then
    append_stage_json "${stage_name}" "PASS" "${elapsed_s}" "${log_file}" "${cmd_text}"
    log_line "STAGE PASS: ${stage_name} (${elapsed_s}s)"
    print_stage_result "PASS" "${stage_name}"
  else
    append_stage_json "${stage_name}" "FAIL" "${elapsed_s}" "${log_file}" "${cmd_text}"
    OVERALL_STATUS="FAIL"
    if [[ -z "${FAILED_STAGE}" ]]; then
      FAILED_STAGE="${stage_name}"
    fi
    log_line "STAGE FAIL: ${stage_name} (${elapsed_s}s) log=${log_file}"
    print_stage_result "FAIL" "${stage_name}"
    if [[ "${FAIL_FAST}" == "1" ]]; then
      return 1
    fi
    return 0
  fi
}

run_optional_playwright_stage() {
  local stage_name="$1"
  local test_cmd="$2"
  local log_file="${LOG_DIR}/${stage_name}.log"
  local stage_prefix stage_num

  stage_prefix="${stage_name%%_*}"
  if [[ "${stage_prefix}" =~ ^([0-9]+) ]]; then
    stage_num=$((10#${BASH_REMATCH[1]}))
  else
    echo "ERROR: stage name does not start with numeric prefix: ${stage_name}" >&2
    return 2
  fi

  if (( stage_num < START_AT )); then
    append_stage_json "${stage_name}" "SKIP_START_AT" "0" "${log_file}" "${test_cmd}"
    log_line "STAGE SKIP: ${stage_name} (start_at=${START_AT})"
    return 0
  fi
  if [[ "${TESTS_FILTER_ACTIVE}" == "1" && -z "${TESTS_FILTER_NUMS[${stage_num}]+x}" ]]; then
    append_stage_json "${stage_name}" "SKIP_FILTER" "0" "${log_file}" "${test_cmd}"
    log_line "STAGE SKIP: ${stage_name} (tests filter)"
    return 0
  fi
  if [[ "${TESTS_FILTER_ACTIVE}" == "1" ]]; then
    TESTS_FILTER_MATCHED["${stage_num}"]=1
  fi

  log_line "STAGE START: ${stage_name}"
  log_line "COMMAND: ${test_cmd}"
  printf '%s\n' "${test_cmd}" >> "${COMMANDS_TXT}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    append_stage_json "${stage_name}" "SKIP_DRY_RUN" "0" "${log_file}" "${test_cmd}"
    return 0
  fi

  if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("playwright.sync_api") else 1)
PY
  then
    echo "playwright.sync_api not available; browser smoke skipped" > "${log_file}"
    append_stage_json "${stage_name}" "SKIP_NO_PLAYWRIGHT" "0" "${log_file}" "${test_cmd}"
    log_line "STAGE SKIP: ${stage_name} (playwright unavailable)"
    return 0
  fi

  run_stage "${stage_name}" "${test_cmd}"
}

log_line "Output directory: ${OUT_DIR}"
log_line "Runs directory: ${RUNS_DIR}"
log_line "Repository root: ${REPO_ROOT}"
log_line "Fail-fast mode: ${FAIL_FAST}"
log_line "Start-at stage: ${START_AT}"
if [[ "${TESTS_FILTER_ACTIVE}" == "1" ]]; then
  log_line "Tests filter: ${TESTS_FILTER_RAW}"
fi

run_stage "01_static_rng_scan" "python -m tools.diagnostics.static_rng_scan --repo_root . --json"
run_stage "02_verify_p0a_gate" "tools/diagnostics/verify_p0a_determinism_gate.sh --with-tests"
run_stage "03_golden_quick" "./tests/determinism/golden_references/run_golden_suite.py quick"
run_stage "04_golden_standard" "./tests/determinism/golden_references/run_golden_suite.py standard"
run_stage "05_golden_nightly" "./tests/determinism/golden_references/run_golden_suite.py nightly"

# Determinism core runtime guards.
run_stage "06_pytest_determinism_runtime_guards" "pytest -q tests/determinism/test_static_rng_scan.py tests/determinism/test_checkpoint_rng_authority.py tests/determinism/test_no_rng_islands_runtime.py"

# Evolvability phase regression stacks.
# Keep deterministic phase-A mechanics under strict env, but exclude intent-validation
# scenarios which are executed separately with isolated (non-strict) env below.
run_stage "07_pytest_phase_a" "pytest -q tests/evolvability/phase_a --ignore=tests/evolvability/phase_a/test_chemistry_intent_validation.py"
run_stage "08_pytest_phase_b" "env -u EIEL_DETERMINISM_STRICT -u EIEL_STRICT_DETERMINISM -u DEBUG_DETERMINISM pytest -q tests/evolvability/phase_b"
run_stage "09_pytest_phase_c" "pytest -q tests/evolvability/phase_c"
run_stage "10_pytest_phase_d" "pytest -q tests/evolvability/phase_d"
run_stage "11_pytest_phase_e" "pytest -q tests/evolvability/phase_e"
run_stage "12_pytest_phase_chemistry_pkg" "pytest -q tests/evolvability/chemistry"

# Intent validation suites with known historical flake risk.
# Run phase-A intent suite outside strict determinism env to avoid collapsing
# scenario-effect signals under strict guardrail instrumentation.
run_stage "13_pytest_phase_a_intent_validation" "env -u EIEL_DETERMINISM_STRICT -u EIEL_STRICT_DETERMINISM -u DEBUG_DETERMINISM pytest -q tests/evolvability/phase_a/test_chemistry_intent_validation.py"
run_stage "14_pytest_phase_b_intent_validation" "env -u EIEL_DETERMINISM_STRICT -u EIEL_STRICT_DETERMINISM -u DEBUG_DETERMINISM pytest -q tests/evolvability/phase_b/test_modular_genome_intent_validation.py"

# Metrics and branching integration coverage.
run_stage "15_pytest_metrics_integration" "pytest -q tests/integration/metrics/test_epi_engine.py tests/integration/metrics/test_epi_timeseries_writer.py tests/integration/metrics/test_adaptation_velocity.py"
run_stage "16_pytest_branching_determinism" "pytest -q tests/integration/branching/test_branching_determinism.py"

# Control-plane/WebUI regression coverage.
# Exclude environment-dependent TTY allocation test from blocking stage;
# run it separately as an optional diagnostic to avoid false negatives on
# runners that cannot provide PTY semantics consistently.
run_stage "17_pytest_control_plane" "pytest -q tests/control_plane tests/control_plane/webui --deselect=tests/control_plane/webui/test_webui_codex_runtime_scaffold.py::test_codex_runtime_oneshot_tty_required_command_supported"
run_stage "17a_pytest_control_plane_tty_diag" "pytest -q tests/control_plane/webui/test_webui_codex_runtime_scaffold.py::test_codex_runtime_oneshot_tty_required_command_supported || true"
run_optional_playwright_stage "17b_webui_playwright_smoke" "pytest -q tests/control_plane/webui/test_webui_agents_page_d1.py::test_agents_tabs_mobile_layout_playwright_smoke"

# Run-path + API contract coverage for recent run-root and web control changes.
run_stage "17c_pytest_webui_runpath_contracts" "pytest -q tests/control_plane/webui/test_webui_run_discovery_wrappers.py tests/control_plane/webui/test_webui_run_index_resolution.py tests/control_plane/webui/test_webui_builder_launch.py::test_managed_launch_plan_wrapper_root_uses_runs_root tests/control_plane/webui/test_webui_builder_launch.py::test_builder_launch_manifest_mode_returns_provenance_fields tests/control_plane/webui/test_agents_orchestrator_cli.py tests/control_plane/webui/test_webui_agents_orchestration.py"

# Phase 5b trait-injection integration.
run_stage "18_pytest_phase5b_trait_injection" "pytest -q tests/integration/phase5b_e2e/test_trait_injection.py tests/integration/phase5b_e2e/test_trait_injection_population_integration.py"

# Centralized regression contracts (schema + checkpoint content).
run_stage "18b_pytest_regression_contracts" "pytest -q tests/regression"

# Negative/failure-path contracts.
run_stage "18c_pytest_negative_contracts" "pytest -q tests/evolvability/phase_e/test_e1_metrics_engine.py::test_e1_budget_mismatch_detects_rng_drift tests/determinism/p0/test_p0e_rng_budget_enforcement.py::test_p0e_strict_failure_on_spawn_budget_drift tests/control_plane/webui/test_webui_ledger_actions.py::test_ledger_action_disallows_non_allowlisted_script"

# Resume verifier unit/integration checks.
run_stage "19_pytest_phase3_resume_verify" "pytest -q tests/harnesses/phase3_resume/verify/test_phase3_checkpoint_replay.py tests/harnesses/phase3_resume/verify/test_runner_bank_token_mismatch_resync.py"

# Safety net over full suite before replay harness.
# Run in neutral env to avoid strict-mode false negatives already covered
# by dedicated strict stages, and deselect known environment-dependent TTY test.
run_stage "20_pytest_full" "env -u EIEL_DETERMINISM_STRICT -u EIEL_STRICT_DETERMINISM -u DEBUG_DETERMINISM pytest -q --ignore=tests/control_plane/webui/test_webui_builder_launch.py --ignore=tests/control_plane/webui/test_webui_codex_runtime_scaffold.py --deselect=tests/determinism/test_static_rng_scan.py::test_p0a_static_rng_scan_passes"

if [[ "${SKIP_REPLAY}" == "1" ]]; then
  append_stage_json "21_strict_replay_e2e" "SKIP_USER" "0" "${LOG_DIR}/21_strict_replay_e2e.log" "SKIPPED (--skip-replay)"
  append_stage_json "22_evolvability_longrun_multi_resume" "SKIP_USER" "0" "${LOG_DIR}/22_evolvability_longrun_multi_resume.log" "SKIPPED (--skip-replay)"
else
  run_stage "21_strict_replay_e2e" \
    "PYTHONHASHSEED=${PYTHONHASHSEED} EIEL_FORENSIC_V4=${EIEL_FORENSIC_V4} EIEL_POLICY_EXEC_PROBE=${EIEL_POLICY_EXEC_PROBE} EIEL_DETERMINISM_STRICT=${EIEL_DETERMINISM_STRICT} EIEL_STRICT_DETERMINISM=${EIEL_STRICT_DETERMINISM} DEBUG_DETERMINISM=${DEBUG_DETERMINISM} CLEAN_RUNS=1 RUNS_DIR='${RUNS_DIR}' ZIP_OUT_DIR='${RUNS_DIR}' ./tests/harnesses/phase3_resume/checkpoint_resume_determinism_e2e.sh --membership-trace --membership-keep 64 --membership-harness --forensic-membership-profile --resume-gen 10 --resume-gens 10,20 -- --knob sim_cfg.steps_per_generation=8 --knob sim_cfg.phase_e_e1_enable=1 --knob sim_cfg.phase_e_e1_cadence_gen=1 --knob sim_cfg.phase_e_e1_knockout_every_gen=1"
  run_stage "22_evolvability_longrun_multi_resume" \
    "RUNS_DIR='${RUNS_DIR}' tests/evolvability/evolvability_longrun_checkpoint_resume.sh --out-dir '${OUT_DIR}/longrun' --seeds 42 --resume-gens 10,20 --steps-per-generation 8"
fi

if [[ "${TESTS_FILTER_ACTIVE}" == "1" ]]; then
  MISSING_FILTER_NUMS=()
  for __k in "${!TESTS_FILTER_NUMS[@]}"; do
    if [[ -z "${TESTS_FILTER_MATCHED[${__k}]+x}" ]]; then
      MISSING_FILTER_NUMS+=("${__k}")
    fi
  done
  if [[ "${#MISSING_FILTER_NUMS[@]}" -gt 0 ]]; then
    missing_csv="$(IFS=,; echo "${MISSING_FILTER_NUMS[*]}")"
    log_line "ERROR: requested stage numbers not found in this script: ${missing_csv}"
    exit 2
  fi
fi

if [[ "${OVERALL_STATUS}" == "PASS" ]]; then
  log_line "ALL STRONG REGRESSION STAGES PASSED"
  exit 0
fi

log_line "STRONG REGRESSION COMPLETED WITH FAILURES (first failed stage: ${FAILED_STAGE})"
exit 1
