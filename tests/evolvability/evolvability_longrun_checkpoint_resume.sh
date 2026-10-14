#!/usr/bin/env bash
set -euo pipefail

# Evolvability long-run checkpoint/resume harness.
# Wraps strict replay harness and adds direct artifact parity checks.
#
# Defaults:
# - Underlying harness total generations: GEN_TOTAL=22 (unless caller overrides env var).
# - Resume checkpoints: 10,20 (default), yielding 3 runs total:
#     1) fresh Run A to GEN_TOTAL
#     2) resumed Run B from checkpoint gen 10
#     3) resumed Run B from checkpoint gen 20
# - Uses Run A bundles directly (no legacy Run-Z creators by default).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

UTC_NOW="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR_DEFAULT="${REPO_ROOT}/runtime/logs/validation/evolvability/evolvability_longrun_${UTC_NOW}"
OUT_DIR="${OUT_DIR_DEFAULT}"

SEEDS_CSV="42"
RESUME_GEN=10
RESUME_GENS=10,20
MEMBERSHIP_KEEP=64
STEPS_PER_GENERATION=8
EXTRA_KNOBS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --out-dir requires a path" >&2; exit 2; }
      OUT_DIR="$2"
      shift 2
      ;;
    --seeds)
      [[ $# -ge 2 ]] || { echo "ERROR: --seeds requires csv values" >&2; exit 2; }
      SEEDS_CSV="$2"
      shift 2
      ;;
    --resume-gen)
      [[ $# -ge 2 ]] || { echo "ERROR: --resume-gen requires an integer" >&2; exit 2; }
      RESUME_GEN="$2"
      shift 2
      ;;
    --resume-gens)
      [[ $# -ge 2 ]] || { echo "ERROR: --resume-gens requires an integer" >&2; exit 2; }
      RESUME_GENS="$2"
      shift 2
      ;;
    --membership-keep)
      [[ $# -ge 2 ]] || { echo "ERROR: --membership-keep requires an integer" >&2; exit 2; }
      MEMBERSHIP_KEEP="$2"
      shift 2
      ;;
    --steps-per-generation)
      [[ $# -ge 2 ]] || { echo "ERROR: --steps-per-generation requires an integer" >&2; exit 2; }
      STEPS_PER_GENERATION="$2"
      shift 2
      ;;
    --extra-knob)
      [[ $# -ge 2 ]] || { echo "ERROR: --extra-knob requires key=value" >&2; exit 2; }
      EXTRA_KNOBS+=("$2")
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: tests/evolvability/evolvability_longrun_checkpoint_resume.sh [options]

Options:
  --out-dir <path>                Output artifact directory
  --seeds <csv>                   Seed list (default: 42)
  --resume-gen <int>              Checkpoint generation (default: 10)
  --resume-gens <csv>             Resume checkpoints list (default: 10,20)
  --membership-keep <int>         Membership keep cap (default: 64)
  --steps-per-generation <int>    sim_cfg.steps_per_generation (default: 8)
  --extra-knob <key=value>        Additional knob passed to harness (repeatable)

Behavior:
  - Underlying harness default GEN_TOTAL is 22 (override with env var GEN_TOTAL=...).
  - Default execution runs 3 campaigns: A fresh + resumes from gen 10 and 20.
  - Runs strict checkpoint/resume harness per seed.
  - Requires harness equivalence PASS.
  - Performs direct artifact parity checks on:
      phase_e_e1_latest.json
      metrics/phase_e_e1_metrics.jsonl (post-resume window)
      metrics/phase_e_e1_sentinel.jsonl (post-resume window)
      sentinel RNG deltas all zero
      bundle_gen<resume_gen>/metrics_state.json contains phase_e_e1_state
  - Emits stage logs plus summary JSON/TXT.
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
SUMMARY_JSON="${OUT_DIR}/evolvability_longrun_summary.json"
SUMMARY_TXT="${OUT_DIR}/evolvability_longrun_summary.txt"
mkdir -p "${LOG_DIR}"
: > "${STAGE_JSONL}"

export PYTHONHASHSEED=0
export EIEL_FORENSIC_V4=1
export EIEL_POLICY_EXEC_PROBE=1
export EIEL_DETERMINISM_STRICT=1
export EIEL_STRICT_DETERMINISM=1
export DEBUG_DETERMINISM=1
export ZIP_OUT_DIR=runtime/adhoc
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
    "schema_version": "eiel_evolvability_longrun_summary_v1",
    "overall_status": overall_status,
    "failed_stage": failed_stage or None,
    "out_dir": str(out_dir),
    "env": {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
        "EIEL_DETERMINISM_STRICT": os.environ.get("EIEL_DETERMINISM_STRICT", ""),
        "EIEL_STRICT_DETERMINISM": os.environ.get("EIEL_STRICT_DETERMINISM", ""),
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
  local start_s elapsed_s rc
  start_s="${SECONDS}"
  log_line "STAGE START: ${stage_name}"
  log_line "COMMAND: ${cmd_text}"
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

parse_parity_and_assert() {
  local harness_log="$1"
  local parity_json="$2"
  local target_resume_gen="$3"
  python - "$harness_log" "$parity_json" "$target_resume_gen" <<'PY'
import json
import sys
from pathlib import Path

harness_log = Path(sys.argv[1])
parity_json = Path(sys.argv[2])
target_resume_gen = int(sys.argv[3])
text = harness_log.read_text(encoding="utf-8", errors="replace")

run_a = None
run_b = None
archive = None
equivalence_pass = False
verifier_exit_zero = False
for ln in text.splitlines():
    if ln.startswith("[INFO] Run A: "):
        run_a = ln.split(": ", 1)[1].strip()
    if ln.startswith(f"[INFO] Run B(resume_gen={target_resume_gen}): "):
        run_b = ln.split(": ", 1)[1].strip()
    if ln.startswith("[INFO] Archive: "):
        archive = ln.split(": ", 1)[1].strip()
    if "RESULT: PASS" in ln:
        equivalence_pass = True
    if ln.startswith(f"[INFO] Verifier exit (resume_gen={target_resume_gen}): ") and ln.rstrip().endswith(": 0"):
        verifier_exit_zero = True

if not run_a or not run_b:
    raise SystemExit(f"missing Run A / Run B for resume_gen={target_resume_gen} in harness log")

def read_jsonl(path: Path):
    out = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s:
            out.append(json.loads(s))
    return out

ra = Path(run_a)
rb = Path(run_b)

la = json.loads((ra / "phase_e_e1_latest.json").read_text(encoding="utf-8")) if (ra / "phase_e_e1_latest.json").exists() else None
lb = json.loads((rb / "phase_e_e1_latest.json").read_text(encoding="utf-8")) if (rb / "phase_e_e1_latest.json").exists() else None
ma = [r for r in read_jsonl(ra / "metrics" / "phase_e_e1_metrics.jsonl") if int(r.get("generation", -1)) > target_resume_gen]
mb = [r for r in read_jsonl(rb / "metrics" / "phase_e_e1_metrics.jsonl") if int(r.get("generation", -1)) > target_resume_gen]
sa = [r for r in read_jsonl(ra / "metrics" / "phase_e_e1_sentinel.jsonl") if int(r.get("generation", -1)) > target_resume_gen]
sb = [r for r in read_jsonl(rb / "metrics" / "phase_e_e1_sentinel.jsonl") if int(r.get("generation", -1)) > target_resume_gen]

bundle = ra / "checkpoint_bundles_v2" / f"bundle_gen{target_resume_gen:06d}" / "metrics_state.json"
ms = json.loads(bundle.read_text(encoding="utf-8")) if bundle.exists() else {}

parity = {
    "schema_version": "eiel_evolvability_longrun_parity_v1",
    "resume_gen": target_resume_gen,
    "archive": archive,
    "run_a": str(ra),
    "run_b": str(rb),
    "harness_equivalence_pass": equivalence_pass,
    "harness_verifier_exit_zero": verifier_exit_zero,
    "latest_equal": la == lb,
    "post_resume_metrics_rows_equal": ma == mb,
    "post_resume_sentinel_rows_equal": sa == sb,
    "post_resume_sentinel_rng_delta_all_zero_a": all(all(int(v) == 0 for v in (row.get("rng_indices_delta", {}) or {}).values()) for row in sa),
    "post_resume_sentinel_rng_delta_all_zero_b": all(all(int(v) == 0 for v in (row.get("rng_indices_delta", {}) or {}).values()) for row in sb),
    "bundle_metrics_state_exists": bundle.exists(),
    "bundle_phase_e_e1_state_present": isinstance(ms, dict) and isinstance(ms.get("phase_e_e1_state"), dict),
}
parity_json.write_text(json.dumps(parity, sort_keys=True, indent=2) + "\n", encoding="utf-8")

required = [
    parity["harness_equivalence_pass"],
    parity["harness_verifier_exit_zero"],
    parity["latest_equal"],
    parity["post_resume_metrics_rows_equal"],
    parity["post_resume_sentinel_rows_equal"],
    parity["post_resume_sentinel_rng_delta_all_zero_a"],
    parity["post_resume_sentinel_rng_delta_all_zero_b"],
    parity["bundle_metrics_state_exists"],
    parity["bundle_phase_e_e1_state_present"],
]
if not all(required):
    raise SystemExit("direct parity assertions failed; see parity json")
PY
}

IFS=',' read -r -a SEEDS <<<"${SEEDS_CSV}"
IFS=',' read -r -a RESUME_TARGETS <<<"${RESUME_GENS}"
FIRST=1

log_line "Output directory: ${OUT_DIR}"
log_line "Repository root: ${REPO_ROOT}"
log_line "Seeds: ${SEEDS_CSV}"

for raw_seed in "${SEEDS[@]}"; do
  seed="$(echo "${raw_seed}" | xargs)"
  [[ -n "${seed}" ]] || continue
  stage_prefix="seed_${seed}"
  harness_log="${LOG_DIR}/${stage_prefix}__harness.log"
  knob_args=(
    "--knob" "sim_cfg.steps_per_generation=${STEPS_PER_GENERATION}"
    "--knob" "sim_cfg.seed=${seed}"
    "--knob" "sim_cfg.phase_e_e1_enable=1"
    "--knob" "sim_cfg.phase_e_e1_cadence_gen=1"
    "--knob" "sim_cfg.phase_e_e1_knockout_every_gen=1"
  )

  for k in "${EXTRA_KNOBS[@]}"; do
    knob_args+=("--knob" "${k}")
  done

  clean_runs=0
  if [[ "${FIRST}" -eq 1 ]]; then
    clean_runs=1
    FIRST=0
  fi

  cmd=(
    "PYTHONHASHSEED=0"
    "EIEL_FORENSIC_V4=1"
    "EIEL_POLICY_EXEC_PROBE=1"
    "EIEL_DETERMINISM_STRICT=1"
    "EIEL_STRICT_DETERMINISM=1"
    "DEBUG_DETERMINISM=1"
    "CLEAN_RUNS=${clean_runs}"
    "ZIP_OUT_DIR=runtime/adhoc"
    "./tests/harnesses/phase3_resume/checkpoint_resume_determinism_e2e.sh"
    "--membership-trace"
    "--membership-keep" "${MEMBERSHIP_KEEP}"
    "--membership-harness"
    "--resume-gen" "${RESUME_GEN}"
    "--resume-gens" "${RESUME_GENS}"
    "--"
    "${knob_args[@]}"
  )

  run_stage "${stage_prefix}__harness" "$(printf '%q ' "${cmd[@]}")"

  # Execute direct parity assertions for each resume checkpoint.
  for rg_raw in "${RESUME_TARGETS[@]}"; do
    rg="$(echo "${rg_raw}" | xargs)"
    [[ -n "${rg}" ]] || continue
    parity_json="${OUT_DIR}/${stage_prefix}__direct_parity_resume_${rg}.json"
    parity_log="${LOG_DIR}/${stage_prefix}__direct_parity_resume_${rg}.log"
    stage_name="${stage_prefix}__direct_parity_resume_${rg}"

    start_s="${SECONDS}"
    set +e
    parse_parity_and_assert "${harness_log}" "${parity_json}" "${rg}" > "${parity_log}" 2>&1
    rc=$?
    set -e
    elapsed_s="$((SECONDS - start_s))"
    if [[ "${rc}" -eq 0 ]]; then
      append_stage_json "${stage_name}" "PASS" "${elapsed_s}" "${parity_log}" "parse_parity_and_assert ${harness_log} ${parity_json} ${rg}"
      log_line "STAGE PASS: ${stage_name} (${elapsed_s}s)"
    else
      append_stage_json "${stage_name}" "FAIL" "${elapsed_s}" "${parity_log}" "parse_parity_and_assert ${harness_log} ${parity_json} ${rg}"
      OVERALL_STATUS="FAIL"
      FAILED_STAGE="${stage_name}"
      log_line "STAGE FAIL: ${stage_name} (${elapsed_s}s)"
      exit 1
    fi
  done
done

log_line "ALL STAGES PASS"
exit 0
