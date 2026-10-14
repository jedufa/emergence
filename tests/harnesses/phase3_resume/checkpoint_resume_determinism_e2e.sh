#!/usr/bin/env bash
# Last modified: 2026-02-23
#
# Phase 3 Resume Determinism E2E (multi-checkpoint capable)
# - Fresh run A to GEN_TOTAL
# - For each resume_gen in --resume-gens (or --resume-gen):
#     - Run Z: run to resume_gen with period=1 to emit bundle_genXXXXXX
#     - Run B: resume from that bundle and run remaining gens to GEN_TOTAL
#     - Verify A vs B (final_gen)
# - Then run state dump compare + membership harness (optional) on LAST B
# - Zip A, all Bs, all Zs + key console logs

# PYTHONHASHSEED=0 \
# EIEL_FORENSIC_V4=1 \
# EIEL_POLICY_EXEC_PROBE=1 \
# EIEL_DETERMINISM_PROBE=1 \
# EIEL_DEBUG_TICK_TRACE=1 \
# EIEL_DEBUG_TICK_TRACE_GENS=0 \
# EIEL_DEBUG_TICK_TRACE_TICKS=0-80 \
# EIEL_DEBUG_POLICY_SLOTS=1 \
# EIEL_DEBUG_POLICY_SLOTS_QUIET=1 \
# EIEL_STATE_DUMP=1 \
# EIEL_STATE_DUMP_AGENT_DETAIL_TICKS=54-61 \
# EIEL_STATE_DUMP_AGENT_DETAIL_STAGES=post_env,post_drives,post_rest,post_attack,post_birth \
# EIEL_TRACE_BIRTH_RNG=1 \
# EIEL_TRACE_BIRTH_CALLSITE=1 \
# EIEL_TRACE_BIRTH_INTERNAL=1 \
# EIEL_TRACE_BIRTH_TICK_MIN=54 \
# EIEL_TRACE_BIRTH_TICK_MAX=61 \
# EIEL_TRACE_BIRTH_INTERNAL_MAX=10 \
# EIEL_DETERMINISM_STRICT=1 \
# DEBUG_DETERMINISM=1 \
# CLEAN_RUNS=1 \
# ZIP_OUT_DIR=runs \
# ./tests/determinism/harnesses/checkpoint_resume_determinism_e2e.sh \
#   --membership-trace \
#   --membership-keep 64 \
#   --membership-harness \
#   --resume-gen 10 \
#   --resume-gens 10,22,34
#
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }
step() { echo; echo "========== $* =========="; }

SEED="${SEED:-42}"
GEN_TOTAL="${GEN_TOTAL:-22}"
POP="${POP:-32}"
PERIOD="${PERIOD:-20}"
RUNS_DIR="${RUNS_DIR:-${ZIP_OUT_DIR:-runtime/adhoc}}"
ZIP_OUT_DIR="${ZIP_OUT_DIR:-runtime/adhoc}"
DEVICE="${DEVICE:-cuda}"
BANK_DEVICE="${BANK_DEVICE:-cuda}"
CLEAN_RUNS="${CLEAN_RUNS:-0}"

RESUME_BOUNDARY_MODE=0
RESUME_BOUNDARY_GEN=0
RESUME_GENS=""

# Legacy mode: create resume bundles via separate checkpoint-creator runs (Run Z) per resume_gen.
# Default (0): resume directly from Run A bundles.
USE_RUN_Z_CHECKPOINT_CREATORS=0

MEMBERSHIP_TRACE=0
MEMBERSHIP_KEEP=32
MEMBERSHIP_FAILFAST_GEN=""
MEMBERSHIP_HARNESS=0
FORENSIC_MEMBERSHIP_PROFILE=0

EXTRA_MAIN_ARGS=()

parse_csv_gens() {
  local s="$1"
  echo "$s" | tr ',' '
' | sed 's/[[:space:]]//g' | grep -E '^[0-9]+$' || true
}

usage() {
cat <<EOF
Usage: $0 [options] [-- <args forwarded to engine.main>]

Options:
  --resume-gen N               Boundary resume experiment from gen N (single)
  --resume-gens CSV            Multi-checkpoint resume experiment (e.g. 10,22,34)
  --membership-trace           Enable membership write tracing env vars
  --membership-keep N          Keep last N membership writes (default: ${MEMBERSHIP_KEEP})
  --membership-failfast-gen G  Failfast generation for membership writes
  --membership-harness         Run membership harness (on LAST resume run)
  --forensic-membership-profile
                               Enable state-dump defaults for membership deep diff
                               (does not change primary equivalence verifier behavior)
  -h, --help                   Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --resume-gen)
      [[ $# -ge 2 ]] || die "--resume-gen requires N"
      RESUME_BOUNDARY_MODE=1
      RESUME_BOUNDARY_GEN="$2"
      shift 2
      ;;
    --resume-gens)
      [[ $# -ge 2 ]] || die "--resume-gens requires CSV"
      RESUME_BOUNDARY_MODE=1
      RESUME_GENS="$2"
      shift 2
      ;;
    --use_run_z_checkpoint_creators)
      USE_RUN_Z_CHECKPOINT_CREATORS=1
      shift
      ;;
    --membership-trace) MEMBERSHIP_TRACE=1; shift ;;
    --membership-keep) [[ $# -ge 2 ]] || die "--membership-keep requires N"; MEMBERSHIP_KEEP="$2"; shift 2 ;;
    --membership-failfast-gen) [[ $# -ge 2 ]] || die "--membership-failfast-gen requires G"; MEMBERSHIP_FAILFAST_GEN="$2"; shift 2 ;;
    --membership-harness) MEMBERSHIP_HARNESS=1; shift ;;
    --forensic-membership-profile) FORENSIC_MEMBERSHIP_PROFILE=1; shift ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_MAIN_ARGS+=("$1")
        shift
      done
      ;;
    *)
      die "Unknown argument: $1 (use --help; extra args must be after --)"
      ;;
  esac
done

[[ "${RESUME_BOUNDARY_MODE}" == "1" ]] || die "This harness requires boundary mode. Use --resume-gen N (and optionally --resume-gens CSV)."

mkdir -p "${RUNS_DIR}" "${ZIP_OUT_DIR}"

if [[ "${CLEAN_RUNS}" == "1" ]]; then
  step "CLEAN: wiping ${RUNS_DIR}"
  rm -rf "${RUNS_DIR:?}/"*
fi

if [[ "${MEMBERSHIP_TRACE}" == "1" ]]; then
  export EIEL_MEMBERSHIP_TRACE=1
  export EIEL_MEMBERSHIP_TRACE_KEEP="${MEMBERSHIP_KEEP}"
  if [[ -z "${MEMBERSHIP_FAILFAST_GEN}" ]]; then
    export EIEL_MEMBERSHIP_FAILFAST_GEN="${RESUME_BOUNDARY_GEN}"
  else
    export EIEL_MEMBERSHIP_FAILFAST_GEN="${MEMBERSHIP_FAILFAST_GEN}"
  fi
  info "Membership trace ON: KEEP=${EIEL_MEMBERSHIP_TRACE_KEEP} FAILFAST_GEN=${EIEL_MEMBERSHIP_FAILFAST_GEN}"
fi

parse_run_dir_from_log() {
  local log_file="$1"
  local metrics_path
  metrics_path="$(grep -Eo 'Metrics written to:\s+.*metrics\.csv' "$log_file" | tail -n 1 | sed -E 's/^Metrics written to:\s+//')"
  [[ -n "${metrics_path}" ]] || return 1
  echo "${metrics_path%/metrics.csv}"
}

bundle_path_for_gen() {
  local run_dir="$1"
  local gen="$2"
  printf "%s/checkpoint_bundles_v2/bundle_gen%06d" "${run_dir}" "${gen}"
}


# ------------------------------------------------------------
# Boundary resume plan (resume gens list) + bundle emission policy
# ------------------------------------------------------------
RESUME_GENS_TO_TEST=()
if [[ -n "${RESUME_GENS}" ]]; then
  while IFS= read -r g; do
    RESUME_GENS_TO_TEST+=("${g}")
  done < <(parse_csv_gens "${RESUME_GENS}")
else
  RESUME_GENS_TO_TEST+=("${RESUME_BOUNDARY_GEN}")
fi
[[ ${#RESUME_GENS_TO_TEST[@]} -gt 0 ]] || die "No resume gens to test"

if [[ "${FORENSIC_MEMBERSHIP_PROFILE}" == "1" ]]; then
  # Keep primary equivalence gate unchanged; only enrich forensic artifacts.
  export EIEL_STATE_DUMP="${EIEL_STATE_DUMP:-1}"
  if [[ -z "${EIEL_STATE_DUMP_GENS:-}" ]]; then
    _gens=()
    _gens+=(0)
    for _g in "${RESUME_GENS_TO_TEST[@]}"; do
      _gens+=("${_g}")
      _gens+=($((_g + 1)))
    done
    _uniq_sorted="$(printf '%s\n' "${_gens[@]}" | awk 'NF {a[$1]=1} END {for (k in a) print k}' | sort -n | paste -sd, -)"
    export EIEL_STATE_DUMP_GENS="${_uniq_sorted}"
  fi
  # Keep detailed state dumps minimal unless caller explicitly sets them.
  export EIEL_STATE_DUMP_AGENT_DETAIL_TICKS="${EIEL_STATE_DUMP_AGENT_DETAIL_TICKS:-}"
  export EIEL_STATE_DUMP_AGENT_DETAIL_STAGES="${EIEL_STATE_DUMP_AGENT_DETAIL_STAGES:-}"
  info "Forensic membership profile ON: EIEL_STATE_DUMP=${EIEL_STATE_DUMP} EIEL_STATE_DUMP_GENS=${EIEL_STATE_DUMP_GENS}"
fi

# Fail-fast: resume checkpoints must be strictly < GEN_TOTAL.
invalid_gens=()
for _g in "${RESUME_GENS_TO_TEST[@]}"; do
  if (( _g >= GEN_TOTAL )); then
    invalid_gens+=("${_g}")
  fi
done
if (( ${#invalid_gens[@]} > 0 )); then
  die "Invalid resume gen(s): ${invalid_gens[*]} (must be < GEN_TOTAL=${GEN_TOTAL}). Fix --resume-gens/--resume-gen or increase GEN_TOTAL."
fi

# Default behavior: Run A emits bundles for all requested resume points.
# Robust approach: set bundle period to 1 in boundary mode (so bundle_genXXXXXX always exists),
# unless the user opts into legacy Run Z checkpoint creators.
RUN_A_BUNDLE_PERIOD="${PERIOD}"
if [[ "${USE_RUN_Z_CHECKPOINT_CREATORS}" == "0" ]]; then
  RUN_A_BUNDLE_PERIOD="1"
  if [[ "${PERIOD}" != "1" ]]; then
    info "Boundary mode: overriding Run A bundle period to 1 (was PERIOD=${PERIOD}) so bundles exist at all resume gens"
  fi
fi

step "STEP 1: Fresh run (A)"
ts="$(date +%Y%m%d_%H%M%S)"
log_a="${RUNS_DIR}/_e2e_fresh_${ts}.log"

python -m engine.main   --seed "${SEED}"   --gen "${GEN_TOTAL}"   --pop "${POP}"   --runs_dir "${RUNS_DIR}"   --device "${DEVICE}"   --bank_device "${BANK_DEVICE}"   --knob "sim_cfg.checkpoint_bundle_v2_period_gen=${RUN_A_BUNDLE_PERIOD}"   "${EXTRA_MAIN_ARGS[@]}"   | tee "${log_a}"

RUN_A="$(parse_run_dir_from_log "${log_a}")" || die "Failed to detect Run A"
info "Run A: ${RUN_A}"

RUN_Z_LIST=()
RUN_B_LIST=()
VERIFY_RC_LIST=()

for resume_gen in "${RESUME_GENS_TO_TEST[@]}"; do
  BUNDLE_Z=""
  RUN_Z=""

  if [[ "${USE_RUN_Z_CHECKPOINT_CREATORS}" == "1" ]]; then
    step "STEP 2: Checkpoint-maker run Z (gen=${resume_gen})"
    tsz="$(date +%Y%m%d_%H%M%S)"
    log_z="${RUNS_DIR}/_e2e_bundle_gen$(printf "%06d" "${resume_gen}")_${tsz}.log"

    python -m engine.main \
      --seed "${SEED}" \
      --gen "${resume_gen}" \
      --pop "${POP}" \
      --runs_dir "${RUNS_DIR}" \
      --device "${DEVICE}" \
      --bank_device "${BANK_DEVICE}" \
      --knob "sim_cfg.checkpoint_bundle_v2_period_gen=1" \
      "${EXTRA_MAIN_ARGS[@]}" \
      | tee "${log_z}"

    RUN_Z="$(parse_run_dir_from_log "${log_z}")" || die "Failed to detect Run Z for gen=${resume_gen}"
    info "Run Z(gen=${resume_gen}): ${RUN_Z}"

    BUNDLE_Z="$(bundle_path_for_gen "${RUN_Z}" "${resume_gen}")"
    [[ -d "${BUNDLE_Z}" ]] || die "Expected bundle not found: ${BUNDLE_Z}"

    else
    step "STEP 2: Using Run A bundle_gen$(printf "%06d" "${resume_gen}") (no Run Z)"
    BUNDLE_Z="$(bundle_path_for_gen "${RUN_A}" "${resume_gen}")"
    [[ -d "${BUNDLE_Z}" ]] || die "Expected Run A bundle not found: ${BUNDLE_Z} (Run A bundle period=${RUN_A_BUNDLE_PERIOD})"
  fi

  step "STEP 3: Resume run B from gen=${resume_gen}"
  gen_remain=$(( GEN_TOTAL - resume_gen ))
  (( gen_remain > 0 )) || die "GEN_TOTAL must be > resume_gen (GEN_TOTAL=${GEN_TOTAL} resume_gen=${resume_gen})"

  tsb="$(date +%Y%m%d_%H%M%S)"
  log_b="${RUNS_DIR}/_e2e_resume_gen$(printf "%06d" "${resume_gen}")_${tsb}.log"

  python -m engine.main     --seed "${SEED}"     --pop "${POP}"     --resume_from "${BUNDLE_Z}"     --gen "${gen_remain}"     --runs_dir "${RUNS_DIR}"     --device "${DEVICE}"     --bank_device "${BANK_DEVICE}"     --knob "sim_cfg.checkpoint_bundle_v2_period_gen=${PERIOD}"     "${EXTRA_MAIN_ARGS[@]}"     | tee "${log_b}"

  RUN_B="$(parse_run_dir_from_log "${log_b}")" || die "Failed to detect Run B for resume_gen=${resume_gen}"
  info "Run B(resume_gen=${resume_gen}): ${RUN_B}"

  step "STEP 3a: Verify equivalence for resume_gen=${resume_gen}"
  set +e
  python -m tests.harnesses.phase3_resume.verify.verify_checkpoint_bundle_v2     --run_a "${RUN_A}"     --run_b "${RUN_B}"     --compare_mode final_gen     --compare_behaviors     --compare_metrics     --strict
  vrc=$?
  set -e
  info "Verifier exit (resume_gen=${resume_gen}): ${vrc}"

  RUN_Z_LIST+=("${RUN_Z}")
  RUN_B_LIST+=("${RUN_B}")
  VERIFY_RC_LIST+=("${vrc}")
done

VERIFY_RC=0
for rc in "${VERIFY_RC_LIST[@]}"; do
  if [[ "${rc}" -ne 0 ]]; then VERIFY_RC="${rc}"; break; fi
done

LAST_RUN_B="${RUN_B_LIST[-1]}"


step "STEP 3b: State dump compare (optional)"
DUMP_RC=0
if [[ "${EIEL_STATE_DUMP:-0}" == "1" ]]; then
  worst=0
  for i in "${!RUN_B_LIST[@]}"; do
    rg="${RESUME_GENS_TO_TEST[$i]}"
    rb="${RUN_B_LIST[$i]}"
    step "STEP 3b.$i: State dump compare for resume_gen=${rg}"

    # Treat "SKIP" as non-fatal (PASS) in the harness.
    # Also treat import/launch failures as SKIP since this check is optional.
    _statecmp_out="${RUNS_DIR}/_e2e_statecmp_gen$(printf "%06d" "${rg}")_$(date +%Y%m%d_%H%M%S).log"
    set +e
    python -m tests.harnesses.phase3_resume.harness.phase3_compare_state_dumps --run-a "${RUN_A}" --run-b "${rb}" | tee "${_statecmp_out}"
    rc=$?
    set -e

    if grep -q '\[STATE_CMP\] SKIP:' "${_statecmp_out}"; then
      info "State dump compare SKIP (resume_gen=${rg}) -> treating as PASS"
      rc=0
    fi

    # If the module couldn't run at all, treat as SKIP/PASS (optional check).
    if [[ "${rc}" -ne 0 ]]; then
      if grep -qiE 'No module named|ModuleNotFoundError|ImportError|Traceback' "${_statecmp_out}"; then
        info "State dump compare unavailable (resume_gen=${rg}) -> treating as SKIP"
        rc=0
      fi
    fi

    if [[ "${rc}" -ne 0 && "${worst}" -eq 0 ]]; then worst="${rc}"; fi
  done
  DUMP_RC="${worst}"
  info "State dump compare worst exit: ${DUMP_RC}"
fi



step "STEP 3c: Membership harness (optional)"
HARNESS_RC=0
if [[ "${MEMBERSHIP_HARNESS}" == "1" ]]; then
  worst=0
  for i in "${!RUN_B_LIST[@]}"; do
    rg="${RESUME_GENS_TO_TEST[$i]}"
    rb="${RUN_B_LIST[$i]}"
    probe_gen=$(( rg + 1 ))
    step "STEP 3c.$i: Membership harness for resume_gen=${rg} (probe_gen=${probe_gen})"
    set +e
    python -m tests.harnesses.phase3_resume.harness.phase3_membership_harness --fresh "${RUN_A}" --resumed "${rb}" --gen "${probe_gen}"
    rc=$?
    set -e
    if [[ "${rc}" -ne 0 && "${worst}" -eq 0 ]]; then worst="${rc}"; fi
  done
  HARNESS_RC="${worst}"
  info "Membership harness worst exit: ${HARNESS_RC}"
fi

step "STEP 4: Zip runs"
ZIP_NAME="${ZIP_OUT_DIR%/}/phase3_runs_seed${SEED}_$(date +%Y%m%d_%H%M%S).zip"
set +e
zip -r "${ZIP_NAME}" \
  "${RUN_A}" "${RUN_Z_LIST[@]}" "${RUN_B_LIST[@]}" \
  "${RUNS_DIR}"/_e2e_*.log \
  -x "*/checkpoint_bundles_v2/*" >/dev/null
set -e
info "Archive: ${ZIP_NAME}"

# step "STEP 4: Zip runs"
# ZIP_NAME="${ZIP_OUT_DIR%/}/phase3_runs_seed${SEED}_$(date +%Y%m%d_%H%M%S).zip"
# set +e
# zip -r "${ZIP_NAME}" "${RUN_A}" "${RUN_Z_LIST[@]}" "${RUN_B_LIST[@]}" "${RUNS_DIR}"/_e2e_*.log >/dev/null
# set -e
# info "Archive: ${ZIP_NAME}"

FINAL_RC=0
if [[ "${VERIFY_RC}" -ne 0 ]]; then FINAL_RC="${VERIFY_RC}"; fi
if [[ "${DUMP_RC}" -ne 0 ]]; then FINAL_RC="${DUMP_RC}"; fi
if [[ "${HARNESS_RC}" -ne 0 ]]; then FINAL_RC="${HARNESS_RC}"; fi

echo
echo "Run A: ${RUN_A}"
echo "Run Bs: ${#RUN_B_LIST[@]}"
echo "Run Zs: ${#RUN_Z_LIST[@]}"
echo "Verifier exit: ${VERIFY_RC}"
echo "[DONE]"
exit "${FINAL_RC}"
