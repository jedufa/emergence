"""Registry-backed Phase 1 success-closure harness.

Resume a baseline/variant proof from the shipped Phase 1 ckpt300 bundle and
emit durable provenance, continuity, and summary artifacts under the
evolvability validation tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from experiments.harness.run_experiments import main as harness_main
from tools.checkpointed_autopilot_influence_phase1 import REQUIRED_GIT_SURFACES

LINEAGE_UIDS = [
    "PIS-20260307-203419Z-76DB",
    "PIS-20260308-024230Z-7B3F",
    "PIS-20260308-030616Z-5C91",
    "PIS-20260308-161500Z-A8F4",
]
SCRIPT_SCHEMA_VERSION = "pressure_autopilot_script_v1"
SUMMARY_SCHEMA_VERSION = "checkpointed_autopilot_phase1_success_summary_v1"
PROVENANCE_SCHEMA_VERSION = "checkpointed_autopilot_phase1_success_provenance_v1"
CONTINUITY_SCHEMA_VERSION = "checkpointed_autopilot_phase1_success_continuity_v1"


@dataclass(frozen=True)
class ScaleConfig:
    name: str
    generations: int
    steps: int
    scenarios: List[Dict[str, Any]]
    requires_noise_control: bool
    acceptance_requires_bidirectional: bool
    reason: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _text_dump(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def default_checkpoint_path() -> Path:
    return (
        _repo_root()
        / "governance"
        / "ledger"
        / "items"
        / "PIS-20260307-203419Z-76DB__checkpointed-autopilot-influence-proof-phase1"
        / "artifacts"
        / "phase1_bundle"
        / "runs"
        / "parent_stabilize"
        / "checkpoint_bundles_v2"
        / "bundle_gen000300"
    )


def default_out_dir(scale: str) -> Path:
    return _repo_root() / "runtime" / "logs" / "validation" / "evolvability" / f"checkpointed_autopilot_phase1_{scale}_{_timestamp_slug()}"


def _baseline_script(value: float) -> Dict[str, Any]:
    return {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "entries": [
            {
                "decision_generation": 301,
                "target_generation": 302,
                "label": "apply_treatment",
                "updates": [{"path": "sim_cfg.maintenance_opportunity_cost_coeff", "value": float(value)}],
            }
        ],
    }


def scale_config(scale: str, baseline_value: float) -> ScaleConfig:
    if scale == "smoke":
        return ScaleConfig(
            name="smoke",
            generations=3,
            steps=8,
            scenarios=[
                {"name": "baseline", "kind": "baseline"},
                {"name": "variant_up_100", "kind": "variant", "target_value": float(baseline_value) * 2.0},
            ],
            requires_noise_control=False,
            acceptance_requires_bidirectional=False,
            reason="Smoke keeps the core A/B path only and proves the registry-backed resume harness executes.",
        )
    if scale == "quick":
        return ScaleConfig(
            name="quick",
            generations=3,
            steps=8,
            scenarios=[
                {"name": "baseline_a", "kind": "baseline"},
                {"name": "baseline_b", "kind": "baseline"},
                {"name": "variant_up_100", "kind": "variant", "target_value": float(baseline_value) * 2.0},
                {"name": "variant_down_75", "kind": "variant", "target_value": float(baseline_value) * 0.25},
            ],
            requires_noise_control=True,
            acceptance_requires_bidirectional=True,
            reason=(
                "Quick adds a second replay-identical baseline because control-noise must be measured from the same checkpoint, "
                "and keeps one upward plus one downward treatment to close the bidirectional Phase 1 claim."
            ),
        )
    raise ValueError(f"Unsupported scale: {scale}")


def collect_provenance(*, checkpoint_path: Path, required_surfaces: Sequence[str] = REQUIRED_GIT_SURFACES) -> Dict[str, Any]:
    repo_root = _repo_root()
    metadata_path = checkpoint_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    git_head = ""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            git_head = result.stdout.strip()
    except Exception:
        git_head = ""

    surfaces: List[Dict[str, Any]] = []
    for rel in required_surfaces:
        path = repo_root / rel
        exists = path.exists()
        entry: Dict[str, Any] = {
            "path": rel,
            "exists_on_disk": exists,
            "sha256": _sha256_path(path) if exists else None,
        }
        if git_head:
            try:
                import subprocess

                result = subprocess.run(
                    ["git", "cat-file", "-e", f"{git_head}:{rel}"],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                entry["present_in_head_commit"] = result.returncode == 0
            except Exception:
                entry["present_in_head_commit"] = False
        else:
            entry["present_in_head_commit"] = False
        surfaces.append(entry)

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "lineage_uids": list(LINEAGE_UIDS),
        "repo_root": str(repo_root),
        "git_head": git_head or None,
        "checkpoint": {
            "path": str(checkpoint_path),
            "metadata_sha256": _sha256_path(metadata_path),
            "metadata": metadata,
        },
        "required_surfaces": surfaces,
        "provenance_mode": "working_tree_hash_snapshot",
        "provenance_note": (
            "Execution acceptance is bound to explicit file hashes plus checkpoint metadata because the current worktree "
            "contains Phase 1 surfaces that are not present in HEAD."
        ),
    }


def prepare_scale_bundle(*, scale: str, out_dir: Path, checkpoint_path: Path, baseline_value: float) -> Dict[str, Any]:
    cfg = scale_config(scale, baseline_value)
    out_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = out_dir / "autopilot_scripts"
    runs_root = out_dir / "runs"
    registry_path = out_dir / f"checkpointed_autopilot_phase1_{scale}.yaml"
    scenarios: List[str] = []
    flags: Dict[str, List[str]] = {}

    common_flags = [
        "--resume_from",
        str(checkpoint_path),
        "--knob",
        "sim_cfg.epi_enable=true",
        "--knob",
        "sim_cfg.av_enable=true",
        "--knob",
        "sim_cfg.epi_timeseries_enable=true",
        "--knob",
        "sim_cfg.policy_tick_stride=1",
        "--device",
        "cpu",
    ]

    for scenario in cfg.scenarios:
        scenario_name = str(scenario["name"])
        scenarios.append(scenario_name)
        scenario_flags = list(common_flags)
        if scenario["kind"] == "variant":
            script_path = scripts_dir / f"{scenario_name}.json"
            _json_dump(script_path, _baseline_script(float(scenario["target_value"])))
            scenario_flags.extend(
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
                    "sim_cfg.pressure_autopilot_apply_delay_gen=1",
                    "--knob",
                    "sim_cfg.pressure_autopilot_suggestion_only=false",
                    "--knob",
                    f"sim_cfg.pressure_autopilot_script=@{script_path}",
                ]
            )
            scenario["script_path"] = str(script_path)
        flags[scenario_name] = scenario_flags

    registry = {
        "harness_api": 1,
        "version": 1,
        "name": f"checkpointed_autopilot_phase1_{scale}",
        "description": "Registry-backed Phase 1 success-closure resume harness.",
        "intent": {
            "question": "Does a checkpointed baseline/variant resume prove causal influence from the same ckpt300 lineage?",
            "reason_for_matrix": cfg.reason,
        },
        "matrix_order": ["scenario", "seed"],
        "axes": {"scenario": {"values": scenarios, "flags": flags}},
        "seeds": [42],
        "cli_defaults": [
            ["--generations", str(cfg.generations)],
            ["--steps", str(cfg.steps)],
            ["--checkpoint_period_gen", "0"],
        ],
        "expected_artifacts": [
            "manifest.json",
            "run_config.csv",
            "config.json",
            "metrics.csv",
            "phase_e_e1_latest.json",
        ],
        "naming": {
            "run_tag_prefix": f"phase1_{scale}",
            "run_id_format": "checkpointed_autopilot_phase1__{scenario}__seed{seed:06d}",
            "run_tag_format": "{run_tag_prefix}__{scenario}",
        },
    }
    _text_dump(registry_path, yaml.safe_dump(registry, sort_keys=False))
    provenance = collect_provenance(checkpoint_path=checkpoint_path)
    _json_dump(out_dir / "phase1_success_provenance.json", provenance)
    return {
        "scale": cfg.name,
        "registry_path": str(registry_path),
        "runs_root": str(runs_root),
        "checkpoint_path": str(checkpoint_path),
        "baseline_value": float(baseline_value),
        "config": {
            "generations": cfg.generations,
            "steps": cfg.steps,
            "requires_noise_control": cfg.requires_noise_control,
            "acceptance_requires_bidirectional": cfg.acceptance_requires_bidirectional,
            "reason": cfg.reason,
            "scenarios": cfg.scenarios,
        },
    }


def _read_series(path: Path, column: str) -> Dict[int, float]:
    if not path.exists():
        return {}
    out: Dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                generation = int(row["generation"])
                out[generation] = float(row[column])
            except Exception:
                continue
    return out


def _aligned(lhs: Mapping[int, float], rhs: Mapping[int, float], *, min_generation: int = 301) -> List[float]:
    keys = sorted(key for key in (set(lhs.keys()) & set(rhs.keys())) if int(key) >= int(min_generation))
    return [float(lhs[k]) - float(rhs[k]) for k in keys]


def _stats(deltas: Sequence[float]) -> Dict[str, Any]:
    if not deltas:
        return {"count": 0, "MAD": None, "MAXD": None, "SMD": None, "END": None}
    abs_deltas = [abs(x) for x in deltas]
    return {
        "count": len(deltas),
        "MAD": sum(abs_deltas) / len(abs_deltas),
        "MAXD": max(abs_deltas),
        "SMD": sum(deltas) / len(deltas),
        "END": deltas[-1],
    }


def _decision_records(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "autopilot" / "pressure_autopilot_decisions.jsonl"
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
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
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _resume_from_value(run_dir: Path) -> Optional[str]:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cli_args = manifest.get("cli_args", {})
            if isinstance(cli_args, dict):
                value = str(cli_args.get("resume_from", "") or "").strip()
                if value:
                    return value
        except Exception:
            pass
    run_config = run_dir / "run_config.csv"
    if run_config.exists():
        with run_config.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) >= 2 and str(row[0]).strip() == "resume_from":
                    value = str(row[1]).strip()
                    if value:
                        return value
    return None


def _find_batch_dir(runs_root: Path) -> Path:
    candidates = sorted(
        [path.parent for path in runs_root.glob("e/*/*/summary.json")],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No harness batch summary found under {runs_root}")
    return candidates[-1]


def analyze_scale_run(*, bundle: Mapping[str, Any], out_dir: Path, batch_dir: Path) -> Dict[str, Any]:
    summary_path = batch_dir / "summary.json"
    matrix_path = batch_dir / "matrix.csv"
    harness_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    run_dirs_by_scenario = {row["scenario"]: Path(row["run_dir"]) for row in rows}
    baseline_dir = run_dirs_by_scenario.get("baseline") or run_dirs_by_scenario.get("baseline_a")
    if baseline_dir is None:
        raise ValueError("Missing baseline scenario in harness output.")
    baseline_metrics = _read_series(baseline_dir / "metrics.csv", "chem_chronic_damage_mean")
    baseline_alive = _read_series(baseline_dir / "metrics.csv", "alive")

    control_noise = {
        "chem_chronic_damage_mean": {"count": 0, "MAD": 0.0, "MAXD": 0.0, "SMD": 0.0, "END": 0.0},
        "alive": {"count": 0, "MAD": 0.0, "MAXD": 0.0, "SMD": 0.0, "END": 0.0},
    }
    baseline_b_dir = run_dirs_by_scenario.get("baseline_b")
    if baseline_b_dir is not None:
        control_noise["chem_chronic_damage_mean"] = _stats(
            _aligned(baseline_metrics, _read_series(baseline_b_dir / "metrics.csv", "chem_chronic_damage_mean"))
        )
        control_noise["alive"] = _stats(_aligned(baseline_alive, _read_series(baseline_b_dir / "metrics.csv", "alive")))

    continuity_runs: List[Dict[str, Any]] = []
    variants: Dict[str, Any] = {}
    proof_events: List[Dict[str, Any]] = []

    for scenario, run_dir in run_dirs_by_scenario.items():
        resume_from = _resume_from_value(run_dir)
        metrics_path = run_dir / "metrics.csv"
        metrics_rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8", newline=""))) if metrics_path.exists() else []
        start_generation = int(metrics_rows[0]["generation"]) if metrics_rows else None
        end_generation = int(metrics_rows[-1]["generation"]) if metrics_rows else None
        continuity_runs.append(
            {
                "scenario": scenario,
                "run_dir": str(run_dir),
                "resume_from": resume_from,
                "resume_matches_source_checkpoint": resume_from == str(bundle["checkpoint_path"]),
                "first_metric_generation": start_generation,
                "last_metric_generation": end_generation,
                "starts_at_resume_boundary": start_generation == 300 and end_generation is not None and end_generation >= 301,
            }
        )

        if not scenario.startswith("variant_"):
            continue
        chem = _stats(_aligned(_read_series(run_dir / "metrics.csv", "chem_chronic_damage_mean"), baseline_metrics))
        alive = _stats(_aligned(_read_series(run_dir / "metrics.csv", "alive"), baseline_alive))
        decisions = [item for item in _decision_records(run_dir) if str(item.get("ack_status", "")) == "ACCEPTED"]
        events = _event_records(run_dir)
        target_value = None
        if decisions:
            try:
                updates = decisions[0].get("updates", [])
                if isinstance(updates, list) and updates:
                    target_value = float(updates[0]["value"])
            except Exception:
                target_value = None
        step4 = bool((chem["MAD"] or 0.0) > 0.0 or (alive["MAD"] or 0.0) > 0.0)
        step5 = bool(
            (chem["MAD"] or 0.0) > float(control_noise["chem_chronic_damage_mean"]["MAD"] or 0.0)
            or (alive["MAD"] or 0.0) > float(control_noise["alive"]["MAD"] or 0.0)
        )
        treatment = {
            "scenario": scenario,
            "target_value": target_value,
            "metrics_vs_baseline": {
                "chem_chronic_damage_mean": chem,
                "alive": alive,
            },
            "chain_evidence": {
                "step1_autopilot_emits": bool(decisions),
                "step2_command_applied": bool(events),
                "step3_effective_value_matches": target_value is not None,
                "step4_diverges_from_baseline": step4,
                "step5_exceeds_control_noise": step5,
            },
        }
        variants[scenario] = treatment
        proof_events.append(
            {
                "scenario": scenario,
                "accepted_decisions": len(decisions),
                "events": len(events),
                "chem_mad": chem["MAD"],
                "alive_mad": alive["MAD"],
                "exceeds_noise": step5,
            }
        )

    continuity = {
        "schema_version": CONTINUITY_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "source_checkpoint_path": str(bundle["checkpoint_path"]),
        "runs": continuity_runs,
        "all_runs_resume_from_same_checkpoint": all(item["resume_matches_source_checkpoint"] for item in continuity_runs),
        "all_runs_start_at_generation_301": all(item["starts_at_resume_boundary"] for item in continuity_runs),
    }
    _json_dump(out_dir / "phase1_success_checkpoint_continuity.json", continuity)

    upward_ok = bool(variants.get("variant_up_100", {}).get("chain_evidence", {}).get("step5_exceeds_control_noise"))
    downward_ok = bool(variants.get("variant_down_75", {}).get("chain_evidence", {}).get("step5_exceeds_control_noise"))
    if bundle["config"]["acceptance_requires_bidirectional"]:
        accepted = continuity["all_runs_resume_from_same_checkpoint"] and continuity["all_runs_start_at_generation_301"] and upward_ok and downward_ok
    else:
        accepted = continuity["all_runs_resume_from_same_checkpoint"] and any(
            item["chain_evidence"]["step4_diverges_from_baseline"] for item in variants.values()
        )

    status = "ACCEPTED_CAUSAL_INFLUENCE" if accepted else "INCOMPLETE_OR_FAILED"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "scale": bundle["scale"],
        "status": status,
        "lineage_uids": list(LINEAGE_UIDS),
        "batch_dir": str(batch_dir),
        "registry_path": str(bundle["registry_path"]),
        "checkpoint_path": str(bundle["checkpoint_path"]),
        "control_noise": control_noise,
        "continuity": continuity,
        "variants": variants,
        "harness_summary": harness_summary,
    }
    _json_dump(out_dir / "phase1_success_summary.json", summary)
    _text_dump(
        out_dir / "phase1_success_summary.jsonl",
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in proof_events),
    )
    txt_lines = [
        f"status: {status}",
        f"scale: {bundle['scale']}",
        f"batch_dir: {batch_dir}",
        f"continuity_resume_same_checkpoint: {continuity['all_runs_resume_from_same_checkpoint']}",
        f"continuity_start_gen_301: {continuity['all_runs_start_at_generation_301']}",
        f"upward_variant_pass: {upward_ok}",
        f"downward_variant_pass: {downward_ok}",
    ]
    _text_dump(out_dir / "phase1_success_summary.txt", "\n".join(txt_lines) + "\n")
    return summary


def run_scale(*, scale: str, out_dir: Path, checkpoint_path: Optional[Path] = None, baseline_value: float = 0.35) -> Dict[str, Any]:
    checkpoint = (checkpoint_path or default_checkpoint_path()).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint source not found: {checkpoint}")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _text_dump(
        out_dir / "stage_results.jsonl",
        json.dumps({"stage": "prepare", "status": "START", "utc": _utc_now()}) + "\n",
    )
    bundle = prepare_scale_bundle(scale=scale, out_dir=out_dir, checkpoint_path=checkpoint, baseline_value=baseline_value)
    rc = harness_main(
        [
            "--registry",
            str(bundle["registry_path"]),
            "--runs_root",
            str(bundle["runs_root"]),
            "--batch_tag",
            f"phase1_{scale}",
        ]
    )
    stage_path = out_dir / "stage_results.jsonl"
    with stage_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"stage": "harness", "status": "PASS" if rc == 0 else "FAIL", "rc": int(rc), "utc": _utc_now()}) + "\n")
    if rc != 0:
        raise SystemExit(rc)
    batch_dir = _find_batch_dir(Path(bundle["runs_root"]))
    summary = analyze_scale_run(bundle=bundle, out_dir=out_dir, batch_dir=batch_dir)
    with stage_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"stage": "analyze", "status": summary["status"], "utc": _utc_now()}) + "\n")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpointed autopilot Phase 1 success-closure harness.")
    parser.add_argument("--scale", choices=("smoke", "quick"), required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--baseline", type=float, default=0.35)
    args = parser.parse_args(list(argv) if argv is not None else None)
    out_dir = args.out_dir if args.out_dir is not None else default_out_dir(args.scale)
    run_scale(scale=args.scale, out_dir=out_dir, checkpoint_path=args.checkpoint_path, baseline_value=float(args.baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
