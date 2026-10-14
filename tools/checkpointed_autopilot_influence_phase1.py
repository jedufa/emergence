"""Phase 1 package for PIS-20260307-203419Z-76DB.

Builds a repo-local execution bundle for the checkpointed autopilot influence
proof and computes comparison outputs after runs complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PIS_UID = "PIS-20260307-203419Z-76DB"
SCRIPT_SCHEMA_VERSION = "pressure_autopilot_script_v1"
PLAN_SCHEMA_VERSION = "checkpointed_autopilot_influence_phase1_run_plan_v1"
COMPARISON_SCHEMA_VERSION = "checkpointed_autopilot_influence_phase1_comparison_v1"
REPORT_SCHEMA_VERSION = "checkpointed_autopilot_influence_phase1_report_v1"
DEFAULT_BASELINE_VALUE = 0.35
WORKFLOW_MAX_CORRECTION_CYCLES = 5
WORKFLOW_METADATA = {
    "max_correction_cycles": WORKFLOW_MAX_CORRECTION_CYCLES,
    "override_note": "Workflow override: maximum correction cycles before escalation is 5.",
    "stop_rule": "If workflow is stopped, write a superseding PIS/state artifact recording completed progress.",
    "retry_rule": "If the final execution run fails, allow one more builder fix/retry when it reduces concrete risk.",
}

REQUIRED_GIT_SURFACES = (
    "tools/checkpointed_autopilot_influence_phase1.py",
    "tests/tools/test_checkpointed_autopilot_influence_phase1.py",
    "tests/simulation/test_pressure_autopilot_scripted.py",
)

RUN_WINDOWS = [
    {"id": "early", "start_generation": 301, "end_generation": 330},
    {"id": "mid", "start_generation": 331, "end_generation": 390},
    {"id": "late", "start_generation": 391, "end_generation": 450},
    {"id": "full", "start_generation": 301, "end_generation": 450},
]

METRIC_SPECS = [
    {
        "id": "alive",
        "source_file": "metrics.csv",
        "column": "alive",
        "description": "Alive population count at generation end.",
    },
    {
        "id": "mean_fitness",
        "source_file": "metrics.csv",
        "column": "mean_fitness",
        "description": "Mean effective fitness from metrics.csv.",
    },
    {
        "id": "chem_chronic_damage_mean",
        "source_file": "metrics.csv",
        "column": "chem_chronic_damage_mean",
        "description": "Chemistry-sensitive burden metric chosen for Phase 1.",
    },
    {
        "id": "successful_births",
        "source_file": "metrics.csv",
        "column": "successful_births",
        "description": "Reproduction-sensitive metric chosen for Phase 1.",
    },
    {
        "id": "EPI",
        "source_file": "epi_av_timeseries.csv",
        "column": "epi_value",
        "description": "Experimental Pressure Index time series.",
    },
    {
        "id": "AV",
        "source_file": "epi_av_timeseries.csv",
        "column": "av_value",
        "description": "Adaptation Velocity time series.",
    },
]

RUN_MATRIX = [
    {"run_id": "parent_stabilize", "category": "parent"},
    {"run_id": "ctrl_resume_a", "category": "control"},
    {"run_id": "ctrl_resume_b", "category": "control"},
    {"run_id": "maint_up_10", "category": "treatment", "mode": "held", "multiplier": 1.10},
    {"run_id": "maint_up_25", "category": "treatment", "mode": "held", "multiplier": 1.25},
    {"run_id": "maint_up_50", "category": "treatment", "mode": "held", "multiplier": 1.50},
    {"run_id": "maint_up_100", "category": "treatment", "mode": "held", "multiplier": 2.00},
    {"run_id": "maint_down_10", "category": "treatment", "mode": "held", "multiplier": 0.90},
    {"run_id": "maint_down_25", "category": "treatment", "mode": "held", "multiplier": 0.75},
    {"run_id": "maint_down_50", "category": "treatment", "mode": "held", "multiplier": 0.50},
    {"run_id": "maint_down_75", "category": "treatment", "mode": "held", "multiplier": 0.25},
    {"run_id": "maint_pulse_up_25_10g", "category": "treatment", "mode": "pulse", "multiplier": 1.25, "revert_generation": 311},
    {"run_id": "maint_pulse_up_50_20g", "category": "treatment", "mode": "pulse", "multiplier": 1.50, "revert_generation": 321},
    {"run_id": "maint_pulse_up_100_40g", "category": "treatment", "mode": "pulse", "multiplier": 2.00, "revert_generation": 341},
    {"run_id": "maint_pulse_down_25_10g", "category": "treatment", "mode": "pulse", "multiplier": 0.75, "revert_generation": 311},
    {"run_id": "maint_pulse_down_50_20g", "category": "treatment", "mode": "pulse", "multiplier": 0.50, "revert_generation": 321},
    {"run_id": "maint_pulse_down_75_40g", "category": "treatment", "mode": "pulse", "multiplier": 0.25, "revert_generation": 341},
]


@dataclass(frozen=True)
class BundlePaths:
    artifact_root: Path
    runs_root: Path
    wrapper_root: Path
    scripts_root: Path
    commands_root: Path
    reports_root: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_git_ref(git_ref: Optional[str]) -> str:
    repo_root = _repo_root()
    candidate = str(git_ref).strip() if git_ref is not None else ""
    rev = "HEAD" if candidate == "" else candidate
    result = subprocess.run(
        ["git", "rev-parse", rev],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"git rev-parse {rev!r} failed"
        raise ValueError(f"Invalid git ref for Phase 1 bundle: {detail}")
    resolved = result.stdout.strip()
    if len(resolved) != 40:
        raise ValueError(f"Resolved git ref is not an immutable 40-hex commit: {resolved}")
    commit_check = subprocess.run(
        ["git", "cat-file", "-e", f"{resolved}^{{commit}}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_check.returncode != 0:
        detail = commit_check.stderr.strip() or commit_check.stdout.strip() or f"git cat-file rejected {resolved}"
        raise ValueError(f"Resolved git ref is not a valid commit: {detail}")
    return resolved


def verify_git_ref_contains_required_surfaces(
    git_ref: str,
    required_paths: Sequence[str] = REQUIRED_GIT_SURFACES,
) -> None:
    repo_root = _repo_root()
    missing: List[str] = []
    for relative_path in required_paths:
        check = subprocess.run(
            ["git", "cat-file", "-e", f"{git_ref}:{relative_path}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode != 0:
            missing.append(str(relative_path))
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(
            "Resolved git ref does not contain required Phase 1 shipped surfaces: "
            f"{joined}"
        )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_executable_text(path: Path, text: str) -> None:
    _atomic_write_text(path, text)
    os.chmod(path, 0o755)


def _paths(artifact_root: Path) -> BundlePaths:
    return BundlePaths(
        artifact_root=artifact_root,
        runs_root=artifact_root / "runs",
        wrapper_root=artifact_root / "wrapper_jobs",
        scripts_root=artifact_root / "autopilot_scripts",
        commands_root=artifact_root / "commands",
        reports_root=artifact_root / "reports",
    )


def _workflow(plan: Dict[str, Any]) -> Dict[str, Any]:
    workflow = plan.get("workflow", {})
    if not isinstance(workflow, dict):
        workflow = {}
    return {
        "max_correction_cycles": int(workflow.get("max_correction_cycles", WORKFLOW_METADATA["max_correction_cycles"])),
        "override_note": str(workflow.get("override_note", WORKFLOW_METADATA["override_note"])),
        "stop_rule": str(workflow.get("stop_rule", WORKFLOW_METADATA["stop_rule"])),
        "retry_rule": str(workflow.get("retry_rule", WORKFLOW_METADATA["retry_rule"])),
    }


def _read_parent_baseline(parent_run_dir: Optional[Path], explicit_baseline: Optional[float]) -> float:
    if explicit_baseline is not None:
        return float(explicit_baseline)
    if parent_run_dir is not None:
        config_path = parent_run_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            sim_cfg = config.get("sim_cfg", {}) if isinstance(config, dict) else {}
            value = sim_cfg.get("maintenance_opportunity_cost_coeff", None)
            if value is not None:
                return float(value)
    return float(DEFAULT_BASELINE_VALUE)


def _series_path(run_dir: Path, metric: Dict[str, Any]) -> Path:
    return run_dir / str(metric["source_file"])


def _run_dir(paths: BundlePaths, run_id: str) -> Path:
    return paths.runs_root / run_id


def _command_string(parts: Iterable[str]) -> str:
    return " ".join(json.dumps(str(part)) for part in parts)


def _script_for_treatment(run: Dict[str, Any], baseline_value: float) -> Dict[str, Any]:
    target_value = float(baseline_value) * float(run["multiplier"])
    entries = [
        {
            "decision_generation": 300,
            "target_generation": 301,
            "label": "apply_treatment",
            "updates": [
                {
                    "path": "sim_cfg.maintenance_opportunity_cost_coeff",
                    "value": float(target_value),
                }
            ],
        }
    ]
    if str(run["mode"]) == "pulse":
        entries.append(
            {
                "decision_generation": int(run["revert_generation"]) - 1,
                "target_generation": int(run["revert_generation"]),
                "label": "revert_to_baseline",
                "updates": [
                    {
                        "path": "sim_cfg.maintenance_opportunity_cost_coeff",
                        "value": float(baseline_value),
                    }
                ],
            }
        )
    return {"schema_version": SCRIPT_SCHEMA_VERSION, "entries": entries}


def build_run_plan(
    *,
    artifact_root: Path,
    seed: int,
    git_ref: str,
    baseline_value: float,
    parent_run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    paths = _paths(artifact_root)
    parent_dir = parent_run_dir or _run_dir(paths, "parent_stabilize")
    checkpoint_path = parent_dir / "checkpoint_bundles_v2" / "bundle_gen000300"
    plan_runs: List[Dict[str, Any]] = []

    for run in RUN_MATRIX:
        run_id = str(run["run_id"])
        run_dir = _run_dir(paths, run_id)
        wrapper_job_dir = paths.wrapper_root / run_id
        entry: Dict[str, Any] = {
            "run_id": run_id,
            "category": str(run["category"]),
            "run_dir": str(run_dir),
            "wrapper_root": str(paths.wrapper_root),
            "wrapper_job_dir": str(wrapper_job_dir),
        }
        if run_id == "parent_stabilize":
            entry.update(
                {
                    "seed": int(seed),
                    "start_generation": 0,
                    "end_generation": 300,
                    "checkpoint_path": str(checkpoint_path),
                    "baseline_value_B": float(baseline_value),
                    "command": parent_command(paths=paths, git_ref=git_ref, seed=seed),
                }
            )
        else:
            entry.update(
                {
                    "seed": int(seed),
                    "start_generation": 301,
                    "end_generation": 450,
                    "resume_from": str(checkpoint_path),
                    "baseline_value_B": float(baseline_value),
                    "command": child_command(
                        run=run,
                        paths=paths,
                        git_ref=git_ref,
                        seed=seed,
                        baseline_value=baseline_value,
                    ),
                }
            )
            if run["category"] == "treatment":
                target_value = float(baseline_value) * float(run["multiplier"])
                entry["mode"] = str(run["mode"])
                entry["target_value"] = float(target_value)
                entry["multiplier"] = float(run["multiplier"])
                script_path = paths.scripts_root / f"{run_id}.json"
                entry["autopilot_script_path"] = str(script_path)
                entry["autopilot_expected_entries"] = _script_for_treatment(run, baseline_value)["entries"]
        plan_runs.append(entry)

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "pis_uid": PIS_UID,
        "git_ref": str(git_ref),
        "seed": int(seed),
        "baseline_value_B": float(baseline_value),
        "checkpoint_generation": 300,
        "checkpoint_path": str(checkpoint_path),
        "intervention_start_generation": 301,
        "run_end_generation": 450,
        "workflow": WORKFLOW_METADATA,
        "metrics": METRIC_SPECS,
        "analysis_windows": RUN_WINDOWS,
        "paths": {
            "artifact_root": str(paths.artifact_root),
            "runs_root": str(paths.runs_root),
            "wrapper_root": str(paths.wrapper_root),
            "scripts_root": str(paths.scripts_root),
            "reports_root": str(paths.reports_root),
        },
        "runs": plan_runs,
    }


def parent_command(*, paths: BundlePaths, git_ref: str, seed: int) -> str:
    repo_root = _repo_root()
    parts = [
        "python",
        "-m",
        "tools.run_job",
        "--module",
        "engine.main",
        "--git_ref",
        git_ref,
        "--job_id",
        "parent_stabilize",
        "--seed",
        str(seed),
        "--wrapper_root",
        str(paths.wrapper_root),
        "--",
        "--runs_dir",
        str(paths.runs_root),
        "--run_id",
        "parent_stabilize",
        "--seed",
        str(seed),
        "--gen",
        "300",
        "--checkpoint_bundle_v2_period_gen",
        "300",
        "--knob",
        "sim_cfg.epi_enable=true",
        "--knob",
        "sim_cfg.av_enable=true",
        "--knob",
        "sim_cfg.epi_timeseries_enable=true",
    ]
    return f"cd {json.dumps(str(repo_root))} && { _command_string(parts) }"


def child_command(
    *,
    run: Dict[str, Any],
    paths: BundlePaths,
    git_ref: str,
    seed: int,
    baseline_value: float,
) -> str:
    repo_root = _repo_root()
    run_id = str(run["run_id"])
    parent_ckpt = _run_dir(paths, "parent_stabilize") / "checkpoint_bundles_v2" / "bundle_gen000300"
    parts = [
        "python",
        "-m",
        "tools.run_job",
        "--module",
        "engine.main",
        "--git_ref",
        git_ref,
        "--job_id",
        run_id,
        "--seed",
        str(seed),
        "--wrapper_root",
        str(paths.wrapper_root),
        "--",
        "--runs_dir",
        str(paths.runs_root),
        "--run_id",
        run_id,
        "--resume_from",
        str(parent_ckpt),
        "--gen",
        "150",
        "--knob",
        "sim_cfg.epi_enable=true",
        "--knob",
        "sim_cfg.av_enable=true",
        "--knob",
        "sim_cfg.epi_timeseries_enable=true",
    ]
    if run["category"] == "treatment":
        script_path = paths.scripts_root / f"{run_id}.json"
        _atomic_write_json(script_path, _script_for_treatment(run, baseline_value))
        parts.extend(
            [
                "--knob",
                "sim_cfg.pressure_autopilot_enable=true",
                "--knob",
                "sim_cfg.pressure_autopilot_profile=scripted",
                "--knob",
                "sim_cfg.pressure_autopilot_tick_interval_gen=1",
                "--knob",
                "sim_cfg.pressure_autopilot_start_gen=300",
                "--knob",
                f"sim_cfg.pressure_autopilot_script=@{script_path}",
            ]
        )
    return f"cd {json.dumps(str(repo_root))} && { _command_string(parts) }"


def write_bundle(
    *,
    artifact_root: Path,
    seed: int,
    git_ref: Optional[str],
    baseline_value: float,
    parent_run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    paths = _paths(artifact_root)
    resolved_git_ref = _resolve_git_ref(git_ref)
    verify_git_ref_contains_required_surfaces(resolved_git_ref)
    for path in (paths.artifact_root, paths.runs_root, paths.wrapper_root, paths.scripts_root, paths.commands_root, paths.reports_root):
        path.mkdir(parents=True, exist_ok=True)

    plan = build_run_plan(
        artifact_root=artifact_root,
        seed=seed,
        git_ref=resolved_git_ref,
        baseline_value=baseline_value,
        parent_run_dir=parent_run_dir,
    )
    _atomic_write_json(paths.artifact_root / "run_plan.json", plan)
    _atomic_write_text(paths.artifact_root / "test_spec.md", render_test_spec(plan))
    _atomic_write_executable_text(
        paths.commands_root / "launch_parent.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n" + parent_command(paths=paths, git_ref=resolved_git_ref, seed=seed) + "\n",
    )
    _atomic_write_executable_text(
        paths.commands_root / "launch_children.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(
            child_command(run=run, paths=paths, git_ref=resolved_git_ref, seed=seed, baseline_value=baseline_value)
            for run in RUN_MATRIX
            if run["category"] != "parent"
        )
        + "\n",
    )
    _atomic_write_json(
        paths.reports_root / "comparison_summary.json",
        {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "pis_uid": PIS_UID,
            "status": "PENDING_RUN_OUTPUTS",
            "windows": plan["analysis_windows"],
            "metrics": plan["metrics"],
            "control_noise": {},
            "treatments": {},
        },
    )
    _atomic_write_json(
        paths.reports_root / "chain_evidence.json",
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "pis_uid": PIS_UID,
            "status": "PENDING_RUN_OUTPUTS",
            "treatments": {},
        },
    )
    report_path = paths.reports_root / "report_scaffold.md"
    _atomic_write_text(report_path, render_report_scaffold(plan))
    return plan


def render_test_spec(plan: Dict[str, Any]) -> str:
    workflow = _workflow(plan)
    lines = [
        f"# {PIS_UID} Phase 1 Test Spec",
        "",
        "## Scope",
        "- In scope: single-lineage causal influence proof for `sim_cfg.maintenance_opportunity_cost_coeff` only.",
        "- Out of scope: policy quality, cross-seed claims, multi-knob optimization, and profile comparisons.",
        "",
        "## Checkpoint Source",
        f"- Parent run: `parent_stabilize`",
        f"- Checkpoint path: `{plan['checkpoint_path']}`",
        f"- Intervention start: generation `301`",
        f"- Child stop: generation `450`",
        f"- Resolved baseline `B`: `{plan['baseline_value_B']:.8f}`",
        f"- Baseline fallback default if checkpoint/config is unavailable: `{DEFAULT_BASELINE_VALUE:.8f}`",
        "",
        "## Metric Mapping",
    ]
    for metric in plan["metrics"]:
        lines.append(f"- `{metric['id']}` -> `{metric['source_file']}:{metric['column']}` ({metric['description']})")
    lines.extend(
        [
            "",
            "## Comparison Formulas",
            "- `MAD`: mean absolute delta between treatment and comparator over aligned generations in a window.",
            "- `MAXD`: maximum absolute delta over the aligned window.",
            "- `SMD`: signed mean delta (`treatment - comparator`) over the aligned window.",
            "- `END`: terminal delta at the window end generation using the last aligned sample in-window.",
            "- Control noise baseline is the same metric/window computation for `ctrl_resume_a` vs `ctrl_resume_b`.",
            "",
            "## Causal Chain Gates",
            "- Step 1: treatment run has at least one `autopilot/pressure_autopilot_decisions.jsonl` record with `ack_status=ACCEPTED`.",
            "- Step 2: corresponding `events/command_events.jsonl` contains the accepted `PressureUpdateCommand` at the planned target generation.",
            "- Step 3: the applied command payload updates `sim_cfg.maintenance_opportunity_cost_coeff` to the plan target value (and pulse reverts to `B` where required).",
            "- Step 4: treatment diverges from `ctrl_resume_a` if any required metric/window has `MAD > 0`.",
            "- Step 5: treatment exceeds resume noise if any required metric/window has `MAD > MAD_noise` or `MAXD > MAXD_noise` or `|END| > |END_noise|`.",
            "",
            "## Pass/Fail Rule",
            "- PASS requires the full 5-step chain for at least one upward treatment and one downward treatment.",
            "- FAIL if the run matrix is incomplete, planned windows/metrics are missing, or no upward/downward treatment clears control-noise.",
            "",
            "## Workflow Governance",
            f"- Maximum correction cycles before escalation: `{workflow['max_correction_cycles']}`.",
            f"- Override note: {workflow['override_note']}",
            f"- Stop rule: {workflow['stop_rule']}",
            f"- Retry rule: {workflow['retry_rule']}",
            "",
            "## Artifact Paths",
            f"- Run plan: `{plan['paths']['artifact_root']}/run_plan.json`",
            f"- Launch scripts: `{plan['paths']['artifact_root']}/commands/launch_parent.sh`, `{plan['paths']['artifact_root']}/commands/launch_children.sh`",
            f"- Comparison outputs: `{plan['paths']['reports_root']}/comparison_summary.json`, `{plan['paths']['reports_root']}/chain_evidence.json`",
            f"- Report scaffold: `{plan['paths']['reports_root']}/report_scaffold.md`",
            "",
            "## Run Matrix",
        ]
    )
    for run in plan["runs"]:
        if run["run_id"] == "parent_stabilize":
            lines.append("- `parent_stabilize`: fresh `0..300`, no intervention, emits `ckpt300`.")
        elif run["category"] == "control":
            lines.append(f"- `{run['run_id']}`: resume `301..450`, no intervention.")
        else:
            target = float(run["target_value"])
            if run["mode"] == "held":
                lines.append(f"- `{run['run_id']}`: held treatment to `{target:.8f}` at generation `301`.")
            else:
                revert_generation = next(item["target_generation"] for item in run["autopilot_expected_entries"] if item["label"] == "revert_to_baseline")
                lines.append(f"- `{run['run_id']}`: pulse to `{target:.8f}` at `301`, revert to `B` at `{revert_generation}`.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _report_lines(
    plan: Dict[str, Any],
    *,
    verdict: str,
    status: str,
    upward_finding: str,
    downward_finding: str,
    resume_noise_finding: str,
    residual_risks: List[str],
) -> List[str]:
    workflow = _workflow(plan)
    lines = [
        f"# {PIS_UID} Phase 1 Report Scaffold",
        "",
        "## Verdict",
        f"- Verdict: {verdict}",
        f"- Status: {status}",
        "",
        "## Evidence Paths",
        f"- Run plan: `{plan['paths']['artifact_root']}/run_plan.json`",
        f"- Test spec: `{plan['paths']['artifact_root']}/test_spec.md`",
        f"- Comparison summary: `{plan['paths']['reports_root']}/comparison_summary.json`",
        f"- Chain evidence: `{plan['paths']['reports_root']}/chain_evidence.json`",
        "",
        "## Findings",
        f"- Upward treatment finding: {upward_finding}",
        f"- Downward treatment finding: {downward_finding}",
        f"- Resume noise finding: {resume_noise_finding}",
        "",
        "## Workflow Governance",
        f"- Maximum correction cycles before escalation: `{workflow['max_correction_cycles']}`.",
        f"- Override note: {workflow['override_note']}",
        f"- Stop rule: {workflow['stop_rule']}",
        f"- Retry rule: {workflow['retry_rule']}",
        "",
        "## Residual Risks",
    ]
    for risk in residual_risks:
        lines.append(f"- {risk}")
    lines.append("")
    return lines


def render_report_scaffold(plan: Dict[str, Any]) -> str:
    return "\n".join(
        _report_lines(
            plan,
            verdict="PRE_RUN",
            status="AWAITING_RUN_OUTPUTS",
            upward_finding="Awaiting treatment outputs.",
            downward_finding="Awaiting treatment outputs.",
            resume_noise_finding="Awaiting control outputs.",
            residual_risks=[
                "Parent stabilization must still be verified on the realized `250..300` window.",
                "Missing run artifacts are handled as FAIL during analysis, not implicit pass.",
            ],
        )
    ) + "\n"


def _read_series(run_dir: Path, metric: Dict[str, Any]) -> Dict[int, Optional[float]]:
    path = _series_path(run_dir, metric)
    if not path.exists():
        return {}
    rows: Dict[int, Optional[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                generation = int(row["generation"])
            except Exception:
                continue
            raw = row.get(str(metric["column"]), "")
            if raw is None or str(raw).strip() == "":
                rows[generation] = None
                continue
            try:
                rows[generation] = float(raw)
            except Exception:
                rows[generation] = None
    return rows


def _window_generations(start_generation: int, end_generation: int) -> range:
    return range(int(start_generation), int(end_generation) + 1)


def _aligned_values(
    lhs: Dict[int, Optional[float]],
    rhs: Dict[int, Optional[float]],
    *,
    start_generation: int,
    end_generation: int,
) -> List[tuple[int, float, float]]:
    aligned: List[tuple[int, float, float]] = []
    for generation in _window_generations(start_generation, end_generation):
        lv = lhs.get(generation, None)
        rv = rhs.get(generation, None)
        if lv is None or rv is None:
            continue
        aligned.append((generation, float(lv), float(rv)))
    return aligned


def _window_stats(aligned: List[tuple[int, float, float]]) -> Dict[str, Any]:
    if not aligned:
        return {"count": 0, "MAD": None, "MAXD": None, "SMD": None, "END": None}
    deltas = [lhs - rhs for _, lhs, rhs in aligned]
    abs_deltas = [abs(delta) for delta in deltas]
    return {
        "count": len(aligned),
        "MAD": sum(abs_deltas) / float(len(abs_deltas)),
        "MAXD": max(abs_deltas),
        "SMD": sum(deltas) / float(len(deltas)),
        "END": deltas[-1],
    }


def _decision_records(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "autopilot" / "pressure_autopilot_decisions.jsonl"
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _event_records(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "events" / "command_events.jsonl"
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def analyze(plan_path: Path) -> Dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    paths = plan["paths"]
    reports_root = Path(paths["reports_root"])
    reports_root.mkdir(parents=True, exist_ok=True)
    runs_by_id = {run["run_id"]: run for run in plan["runs"]}

    control_a = Path(runs_by_id["ctrl_resume_a"]["run_dir"])
    control_b = Path(runs_by_id["ctrl_resume_b"]["run_dir"])
    comparison_summary: Dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "pis_uid": PIS_UID,
        "status": "COMPLETE",
        "windows": plan["analysis_windows"],
        "metrics": plan["metrics"],
        "control_noise": {},
        "treatments": {},
        "missing_run_outputs": [],
    }

    control_series = {metric["id"]: _read_series(control_a, metric) for metric in plan["metrics"]}
    control_b_series = {metric["id"]: _read_series(control_b, metric) for metric in plan["metrics"]}

    for metric in plan["metrics"]:
        metric_id = metric["id"]
        comparison_summary["control_noise"][metric_id] = {}
        for window in plan["analysis_windows"]:
            aligned = _aligned_values(
                control_series[metric_id],
                control_b_series[metric_id],
                start_generation=window["start_generation"],
                end_generation=window["end_generation"],
            )
            comparison_summary["control_noise"][metric_id][window["id"]] = _window_stats(aligned)

    chain_evidence: Dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "pis_uid": PIS_UID,
        "status": "COMPLETE",
        "treatments": {},
    }

    missing_run_outputs: List[str] = []
    for run in plan["runs"]:
        if run["category"] == "parent":
            continue
        run_dir = Path(run["run_dir"])
        if not _has_any_metric_data(run_dir, plan["metrics"]):
            missing_run_outputs.append(str(run["run_id"]))

    for run in plan["runs"]:
        if run["category"] != "treatment":
            continue
        run_id = run["run_id"]
        run_dir = Path(run["run_dir"])
        run_series = {metric["id"]: _read_series(run_dir, metric) for metric in plan["metrics"]}
        comparison_summary["treatments"][run_id] = {}
        for metric in plan["metrics"]:
            metric_id = metric["id"]
            comparison_summary["treatments"][run_id][metric_id] = {}
            for window in plan["analysis_windows"]:
                aligned = _aligned_values(
                    run_series[metric_id],
                    control_series[metric_id],
                    start_generation=window["start_generation"],
                    end_generation=window["end_generation"],
                )
                comparison_summary["treatments"][run_id][metric_id][window["id"]] = _window_stats(aligned)

        decisions = _decision_records(run_dir)
        events = _event_records(run_dir)
        accepted_decisions = [item for item in decisions if str(item.get("ack_status", "")) == "ACCEPTED"]
        expected_entries = run.get("autopilot_expected_entries", [])
        expected_generations = {int(item["target_generation"]): item for item in expected_entries}
        observed_generations = {}
        for event in events:
            payload = event.get("command_payload", {})
            if not isinstance(payload, dict):
                continue
            if str(payload.get("command_type", "")) != "PressureUpdateCommand":
                continue
            generation = int(event.get("generation_applied", -1))
            updates = payload.get("updates", [])
            if not isinstance(updates, list):
                continue
            observed_generations[generation] = updates

        diverges = False
        exceeds_noise = False
        for metric_id, metric_windows in comparison_summary["treatments"][run_id].items():
            for window_id, stats in metric_windows.items():
                noise = comparison_summary["control_noise"][metric_id][window_id]
                if stats["MAD"] is not None and float(stats["MAD"]) > 0.0:
                    diverges = True
                if stats["MAD"] is None or noise["MAD"] is None:
                    continue
                if float(stats["MAD"]) > float(noise["MAD"]):
                    exceeds_noise = True
                if stats["MAXD"] is not None and noise["MAXD"] is not None and float(stats["MAXD"]) > float(noise["MAXD"]):
                    exceeds_noise = True
                if stats["END"] is not None and noise["END"] is not None and abs(float(stats["END"])) > abs(float(noise["END"])):
                    exceeds_noise = True

        chain_evidence["treatments"][run_id] = {
            "step1_autopilot_emits": bool(accepted_decisions),
            "step2_command_applied": bool(observed_generations),
            "step3_effective_value_matches": _check_expected_updates(expected_generations, observed_generations),
            "step4_diverges_from_ctrl_resume_a": bool(diverges),
            "step5_exceeds_ctrl_noise": bool(exceeds_noise),
            "accepted_decisions": len(accepted_decisions),
            "applied_generations": sorted(observed_generations.keys()),
        }

    if missing_run_outputs:
        comparison_summary["status"] = "INCOMPLETE_INPUTS"
        comparison_summary["missing_run_outputs"] = sorted(missing_run_outputs)
        chain_evidence["status"] = "INCOMPLETE_INPUTS"

    _atomic_write_json(reports_root / "comparison_summary.json", comparison_summary)
    _atomic_write_json(reports_root / "chain_evidence.json", chain_evidence)
    _atomic_write_text(reports_root / "report_scaffold.md", synthesize_report(plan, comparison_summary, chain_evidence))
    return {"comparison_summary": comparison_summary, "chain_evidence": chain_evidence}


def _check_expected_updates(expected: Dict[int, Dict[str, Any]], observed: Dict[int, List[Dict[str, Any]]]) -> bool:
    if not expected:
        return False
    for generation, item in expected.items():
        updates = observed.get(int(generation), [])
        if not updates:
            return False
        exp_updates = item.get("updates", [])
        if len(exp_updates) != len(updates):
            return False
        for exp, got in zip(exp_updates, updates):
            if str(exp.get("path", "")) != str(got.get("path", "")):
                return False
            try:
                if float(exp.get("value")) != float(got.get("value")):
                    return False
            except Exception:
                if exp.get("value") != got.get("value"):
                    return False
    return True


def _has_any_metric_data(run_dir: Path, metrics: List[Dict[str, Any]]) -> bool:
    return any(_series_path(run_dir, metric).exists() for metric in metrics)


def _run_all_steps_passed(treatment: Dict[str, Any]) -> bool:
    return all(
        bool(treatment.get(key))
        for key in (
            "step1_autopilot_emits",
            "step2_command_applied",
            "step3_effective_value_matches",
            "step4_diverges_from_ctrl_resume_a",
            "step5_exceeds_ctrl_noise",
        )
    )


def _format_treatment_finding(label: str, matched: List[str], partial: List[str]) -> str:
    if matched:
        return f"{label} PASS via {', '.join(matched)}."
    if partial:
        return f"{label} FAIL; partial evidence in {', '.join(partial)}."
    return f"{label} FAIL; no treatment satisfied the full 5-step chain."


def _format_resume_noise_finding(control_noise: Dict[str, Any], missing_runs: List[str]) -> str:
    if missing_runs:
        return f"FAIL; missing run outputs for {', '.join(missing_runs)}."
    for metric_windows in control_noise.values():
        for stats in metric_windows.values():
            if stats.get("count", 0) > 0:
                return "Control noise computed from ctrl_resume_a vs ctrl_resume_b across all required windows."
    return "FAIL; control runs did not yield aligned metric samples."


def synthesize_report(
    plan: Dict[str, Any],
    comparison_summary: Dict[str, Any],
    chain_evidence: Dict[str, Any],
) -> str:
    missing_runs = list(comparison_summary.get("missing_run_outputs", []))
    upward_pass: List[str] = []
    upward_partial: List[str] = []
    downward_pass: List[str] = []
    downward_partial: List[str] = []
    for run in plan["runs"]:
        if run["category"] != "treatment":
            continue
        run_id = str(run["run_id"])
        evidence = chain_evidence["treatments"].get(run_id, {})
        is_pass = _run_all_steps_passed(evidence)
        if float(run["multiplier"]) > 1.0:
            (upward_pass if is_pass else upward_partial).append(run_id)
        else:
            (downward_pass if is_pass else downward_partial).append(run_id)

    verdict = "PASS" if upward_pass and downward_pass and not missing_runs else "FAIL"
    status = "COMPLETE" if not missing_runs else "INCOMPLETE_INPUTS"
    residual_risks: List[str] = []
    if missing_runs:
        residual_risks.append(f"Missing run outputs: {', '.join(missing_runs)}.")
    residual_risks.append("Parent stabilization lineage validity remains scoped to the realized ckpt300 artifacts.")
    residual_risks.append("This package proves causal influence only; it does not establish policy quality or multi-seed generality.")
    return "\n".join(
        _report_lines(
            plan,
            verdict=verdict,
            status=status,
            upward_finding=_format_treatment_finding("Upward treatment", upward_pass, upward_partial),
            downward_finding=_format_treatment_finding("Downward treatment", downward_pass, downward_partial),
            resume_noise_finding=_format_resume_noise_finding(comparison_summary.get("control_noise", {}), missing_runs),
            residual_risks=residual_risks,
        )
    ) + "\n"


def _default_artifact_root() -> Path:
    return _repo_root() / "governance" / "ledger" / "items" / f"{PIS_UID}__checkpointed-autopilot-influence-proof-phase1" / "artifacts" / "phase1_bundle"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpointed autopilot influence proof Phase 1 bundle tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Write the Phase 1 execution bundle.")
    init_parser.add_argument("--artifact-root", type=Path, default=_default_artifact_root())
    init_parser.add_argument("--seed", type=int, default=42)
    init_parser.add_argument("--git-ref", type=str, default=None)
    init_parser.add_argument("--baseline", type=float, default=None)
    init_parser.add_argument("--parent-run-dir", type=Path, default=None)

    analyze_parser = subparsers.add_parser("analyze", help="Compute comparison outputs from completed runs.")
    analyze_parser.add_argument("--plan", type=Path, default=_default_artifact_root() / "run_plan.json")

    args = parser.parse_args(argv)
    if args.command == "init":
        baseline_value = _read_parent_baseline(args.parent_run_dir, args.baseline)
        write_bundle(
            artifact_root=args.artifact_root,
            seed=int(args.seed),
            git_ref=args.git_ref,
            baseline_value=float(baseline_value),
            parent_run_dir=args.parent_run_dir,
        )
        return 0
    if args.command == "analyze":
        analyze(args.plan)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
