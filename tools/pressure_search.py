"""Controlled pressure-point search supervisor for Emergence.

This is the outer autopilot: it does not change the simulation directly. It
keeps durable campaign state, proposes bounded pressure-autopilot mutations,
and records operator-controlled promotion/rejection decisions.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


CAMPAIGN_SCHEMA_VERSION = "emergence_pressure_search_campaign_v1"
CONTROL_SCHEMA_VERSION = "emergence_pressure_search_control_v1"
EVENT_SCHEMA_VERSION = "emergence_pressure_search_event_v1"
LINEAGE_SCHEMA_VERSION = "emergence_pressure_search_lineage_v1"

DEFAULT_CAMPAIGN_ID = "pressure_search_001"
MUTATION_ALLOWLIST = {
    "sim_cfg.pressure_autopilot_profile",
    "sim_cfg.pressure_autopilot_tick_interval_gen",
    "sim_cfg.pressure_autopilot_delta_up",
    "sim_cfg.pressure_autopilot_delta_down",
    "sim_cfg.pressure_autopilot_target_survival",
    "sim_cfg.pressure_autopilot_survival_ceiling",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_campaigns_root() -> Path:
    return project_root() / "_artifacts" / "pressure_search"


def default_runs_root() -> Path:
    return project_root() / "_artifacts" / "pressure_search_runs"


def default_registry() -> str:
    return "_spec/registry/v1/generated/autopilot_proof_run__a4f3c2__harness.yaml"


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_campaign_id(raw: str) -> str:
    text = str(raw or "").strip() or DEFAULT_CAMPAIGN_ID
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"_", "-"} else "_")
    cleaned = "".join(out).strip("_")
    return cleaned[:80] or DEFAULT_CAMPAIGN_ID


def campaign_dir(campaign_id: str, *, campaigns_root: Optional[Path] = None) -> Path:
    root = Path(campaigns_root or default_campaigns_root()).resolve()
    return root / safe_campaign_id(campaign_id)


def read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists() or not path.is_file() or path.is_symlink():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_event(cdir: Path, event: str, **payload: Any) -> Dict[str, Any]:
    rec = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "utc": now_utc(),
        "event": str(event),
        **payload,
    }
    path = cdir / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
    return rec


def default_control() -> Dict[str, Any]:
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "requested_action": "",
        "requested_utc": "",
        "requested_by": "",
        "target_run_key": "",
        "note": "",
    }


def default_state(campaign_id: str, *, duration_days: int = 28, max_active_children: int = 1) -> Dict[str, Any]:
    start = now_utc()
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": safe_campaign_id(campaign_id),
        "status": "idle",
        "created_utc": start,
        "updated_utc": start,
        "started_utc": "",
        "finished_utc": "",
        "duration_days": int(duration_days),
        "max_active_children": int(max_active_children),
        "profile": "evolvability_band",
        "current_parent_run_key": "",
        "current_parent_score": None,
        "best_run_key": "",
        "best_score": None,
        "active_child_run_key": "",
        "pending_candidates": [],
        "rejected_candidates": [],
        "parked_candidates": [],
        "iteration": 0,
        "consecutive_rejections": 0,
        "last_error": "",
        "last_mutation": {},
        "last_decision": {},
    }


def ensure_campaign(
    campaign_id: str,
    *,
    campaigns_root: Optional[Path] = None,
    duration_days: int = 28,
    max_active_children: int = 1,
) -> Dict[str, Any]:
    cid = safe_campaign_id(campaign_id)
    cdir = campaign_dir(cid, campaigns_root=campaigns_root)
    cdir.mkdir(parents=True, exist_ok=True)
    state_path = cdir / "state.json"
    control_path = cdir / "control.json"
    state = read_json(state_path, None)
    if not isinstance(state, dict) or state.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        state = default_state(cid, duration_days=duration_days, max_active_children=max_active_children)
        write_json(state_path, state)
        append_event(cdir, "campaign_created", campaign_id=cid)
    if not control_path.exists():
        write_json(control_path, default_control())
    _write_summary(cdir, state)
    _write_lineage(cdir, state)
    return state


def list_campaigns(*, campaigns_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    root = Path(campaigns_root or default_campaigns_root()).resolve()
    if not root.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        state = read_json(p / "state.json", {})
        if isinstance(state, dict) and state.get("schema_version") == CAMPAIGN_SCHEMA_VERSION:
            rows.append(_public_campaign_payload(p, state))
    rows.sort(key=lambda r: str(r.get("updated_utc") or ""), reverse=True)
    return rows


def get_campaign(campaign_id: str, *, campaigns_root: Optional[Path] = None) -> Dict[str, Any]:
    cid = safe_campaign_id(campaign_id)
    cdir = campaign_dir(cid, campaigns_root=campaigns_root)
    state = ensure_campaign(cid, campaigns_root=campaigns_root)
    return _public_campaign_payload(cdir, state)


def request_action(
    campaign_id: str,
    action: str,
    *,
    target_run_key: str = "",
    note: str = "",
    requested_by: str = "webui",
    campaigns_root: Optional[Path] = None,
) -> Dict[str, Any]:
    cid = safe_campaign_id(campaign_id)
    state = ensure_campaign(cid, campaigns_root=campaigns_root)
    cdir = campaign_dir(cid, campaigns_root=campaigns_root)
    action_norm = str(action or "").strip().lower()
    if action_norm not in {"start", "pause", "resume", "stop", "promote", "reject"}:
        raise ValueError("action must be start, pause, resume, stop, promote, or reject")
    control = {
        **default_control(),
        "requested_action": action_norm,
        "requested_utc": now_utc(),
        "requested_by": str(requested_by or "webui"),
        "target_run_key": str(target_run_key or "").strip(),
        "note": str(note or "").strip(),
    }
    write_json(cdir / "control.json", control)
    append_event(cdir, "operator_action_requested", action=action_norm, target_run_key=target_run_key, note=note)

    # Apply UI-control actions synchronously so the panel responds immediately.
    if action_norm in {"pause", "resume", "stop", "promote", "reject", "start"}:
        state = apply_control(cdir, state, control)
        write_json(cdir / "state.json", state)
        write_json(cdir / "control.json", default_control())
        _write_lineage(cdir, state)
        _write_summary(cdir, state)
    return _public_campaign_payload(cdir, state)


def apply_control(cdir: Path, state: Dict[str, Any], control: Dict[str, Any]) -> Dict[str, Any]:
    action = str(control.get("requested_action") or "").strip().lower()
    target = str(control.get("target_run_key") or "").strip()
    if not action:
        return state
    state = dict(state)
    if action == "start":
        if state.get("status") in {"idle", "paused", "stopped"}:
            state["status"] = "running"
            state["started_utc"] = state.get("started_utc") or now_utc()
    elif action == "pause":
        if state.get("status") not in {"stopped", "failed"}:
            state["status"] = "paused"
    elif action == "resume":
        if state.get("status") == "paused":
            state["status"] = "running"
    elif action == "stop":
        state["status"] = "stopped"
        state["finished_utc"] = now_utc()
    elif action == "promote":
        state = _promote_candidate(state, target)
    elif action == "reject":
        state = _reject_candidate(state, target)
    state["updated_utc"] = now_utc()
    state["last_decision"] = {"action": action, "target_run_key": target, "utc": state["updated_utc"]}
    append_event(cdir, "operator_action_applied", action=action, target_run_key=target, status=state.get("status"))
    return state


def _candidate_by_run_key(state: Dict[str, Any], run_key: str) -> Optional[Dict[str, Any]]:
    for row in list(state.get("pending_candidates") or []):
        if str(row.get("run_key") or "") == str(run_key):
            return dict(row)
    return None


def _promote_candidate(state: Dict[str, Any], run_key: str) -> Dict[str, Any]:
    candidate = _candidate_by_run_key(state, run_key)
    if candidate is None:
        raise ValueError("candidate not pending for promotion")
    pending = [row for row in list(state.get("pending_candidates") or []) if str(row.get("run_key") or "") != str(run_key)]
    state["pending_candidates"] = pending
    state["current_parent_run_key"] = run_key
    state["current_parent_score"] = candidate.get("score")
    state["best_run_key"] = run_key
    state["best_score"] = candidate.get("score")
    state["active_child_run_key"] = ""
    state["status"] = "running"
    state["consecutive_rejections"] = 0
    return state


def _reject_candidate(state: Dict[str, Any], run_key: str) -> Dict[str, Any]:
    candidate = _candidate_by_run_key(state, run_key)
    if candidate is None:
        raise ValueError("candidate not pending for rejection")
    state["pending_candidates"] = [row for row in list(state.get("pending_candidates") or []) if str(row.get("run_key") or "") != str(run_key)]
    rejected = list(state.get("rejected_candidates") or [])
    rejected.append({**candidate, "rejected_utc": now_utc()})
    state["rejected_candidates"] = rejected[-200:]
    state["status"] = "running"
    state["consecutive_rejections"] = int(state.get("consecutive_rejections") or 0) + 1
    return state


def score_run(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.csv"
    epiav_path = run_dir / "epi_av_timeseries.csv"
    last_metrics = _last_csv_row(metrics_path)
    epi_rows = _csv_rows(epiav_path, limit=2000)
    alive = _num(last_metrics.get("alive"), 0.0)
    survival = _num(last_metrics.get("cohort_survival_rate"), 0.0)
    diversity = _num(last_metrics.get("genome_diversity_norm_std_mean"), 0.0)
    births = _num(last_metrics.get("successful_births"), 0.0)
    collapse_streak = _num(last_metrics.get("diversity_collapse_streak"), 0.0)
    epi_latest = _last_num(epi_rows, "epi_value")
    av_latest = _last_num(epi_rows, "av_value")
    epi_trend = _trend(epi_rows, "epi_value")
    av_trend = _trend(epi_rows, "av_value")
    telemetry_ok = bool(last_metrics) and alive > 0
    score = (
        survival * 35.0
        + min(diversity, 1.0) * 20.0
        + min(births / 50.0, 1.0) * 15.0
        + max(min(epi_trend / 10.0, 1.0), -1.0) * 10.0
        + max(min((av_latest or 0.0) / 5.0, 1.0), -1.0) * 10.0
        + max(min(av_trend / 5.0, 1.0), -1.0) * 10.0
        - min(collapse_streak, 10.0) * 2.0
    )
    if not telemetry_ok:
        score -= 100.0
    return {
        "schema_version": "pressure_search_score_v1",
        "run_dir": str(run_dir),
        "score": round(float(score), 6),
        "telemetry_ok": bool(telemetry_ok),
        "alive": alive,
        "survival": survival,
        "diversity": diversity,
        "successful_births": births,
        "diversity_collapse_streak": collapse_streak,
        "epi_latest": epi_latest,
        "av_latest": av_latest,
        "epi_trend": epi_trend,
        "av_trend": av_trend,
    }


def _csv_rows(path: Path, *, limit: int = 0) -> List[Dict[str, str]]:
    if not path.exists() or not path.is_file() or path.is_symlink():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
            if limit and len(rows) > limit:
                rows = rows[-limit:]
    return rows


def _last_csv_row(path: Path) -> Dict[str, str]:
    rows = _csv_rows(path, limit=1)
    return rows[-1] if rows else {}


def _num(raw: Any, default: float = 0.0) -> float:
    try:
        if raw in (None, ""):
            return float(default)
        return float(raw)
    except Exception:
        return float(default)


def _last_num(rows: List[Dict[str, str]], key: str) -> Optional[float]:
    for row in reversed(rows):
        if str(row.get(key) or "").strip():
            return _num(row.get(key), 0.0)
    return None


def _trend(rows: List[Dict[str, str]], key: str) -> float:
    vals = [_num(row.get(key), 0.0) for row in rows if str(row.get(key) or "").strip()]
    if len(vals) < 2:
        return 0.0
    width = min(20, len(vals) // 2)
    if width <= 0:
        return 0.0
    return float(sum(vals[-width:]) / width) - float(sum(vals[:width]) / width)


def next_mutation(iteration: int) -> Dict[str, Any]:
    cycle = [
        ("sim_cfg.pressure_autopilot_profile", "cycle"),
        ("sim_cfg.pressure_autopilot_tick_interval_gen", 250),
        ("sim_cfg.pressure_autopilot_delta_up", 0.06),
        ("sim_cfg.pressure_autopilot_delta_down", 0.11),
        ("sim_cfg.pressure_autopilot_target_survival", 0.58),
        ("sim_cfg.pressure_autopilot_survival_ceiling", 0.82),
        ("sim_cfg.pressure_autopilot_profile", "balanced"),
        ("sim_cfg.pressure_autopilot_tick_interval_gen", 400),
    ]
    path, value = cycle[int(iteration) % len(cycle)]
    return {
        "schema_version": "pressure_search_mutation_v1",
        "mutation_id": f"m{int(iteration) + 1:05d}",
        "path": path,
        "value": value,
        "knobs": [f"{path}={str(value).lower() if isinstance(value, bool) else value}"],
        "summary": f"Set {path} to {value}",
    }


def discover_completed_runs(runs_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = Path(runs_root).resolve()
    if not root.exists():
        return rows
    for metrics in root.rglob("metrics.csv"):
        run_dir = metrics.parent
        if not run_dir.is_dir():
            continue
        rel = run_dir.relative_to(root).as_posix()
        score = score_run(run_dir)
        rows.append({"run_key": rel, "run_dir": str(run_dir), "score": score.get("score"), "score_detail": score})
    rows.sort(key=lambda row: float(row.get("score") or -999999.0), reverse=True)
    return rows


def choose_initial_parent(runs_root: Path) -> Dict[str, Any]:
    rows = discover_completed_runs(runs_root)
    if rows:
        return rows[0]
    return {"run_key": "", "run_dir": "", "score": None, "score_detail": {}}


def supervisor_step(
    campaign_id: str,
    *,
    runs_root: Optional[Path] = None,
    campaigns_root: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    cid = safe_campaign_id(campaign_id)
    cdir = campaign_dir(cid, campaigns_root=campaigns_root)
    state = ensure_campaign(cid, campaigns_root=campaigns_root)
    control = read_json(cdir / "control.json", default_control())
    if isinstance(control, dict) and control.get("requested_action"):
        state = apply_control(cdir, state, control)
        write_json(cdir / "control.json", default_control())

    if state.get("status") == "idle":
        state["status"] = "running"
        state["started_utc"] = state.get("started_utc") or now_utc()

    if state.get("status") != "running":
        state["updated_utc"] = now_utc()
        write_json(cdir / "state.json", state)
        _write_summary(cdir, state)
        return _public_campaign_payload(cdir, state)

    root = Path(runs_root or default_runs_root()).resolve()
    if not state.get("current_parent_run_key"):
        parent = choose_initial_parent(root)
        state["current_parent_run_key"] = parent.get("run_key") or ""
        state["current_parent_score"] = parent.get("score")
        state["best_run_key"] = parent.get("run_key") or ""
        state["best_score"] = parent.get("score")
        append_event(cdir, "parent_selected", parent=parent)

    if state.get("pending_candidates"):
        state["status"] = "waiting_for_promotion"
    elif not state.get("active_child_run_key"):
        mutation = next_mutation(int(state.get("iteration") or 0))
        state["iteration"] = int(state.get("iteration") or 0) + 1
        state["last_mutation"] = mutation
        child_key = f"dry_run_child_{state['iteration']:05d}" if dry_run else ""
        if dry_run:
            candidate = {
                "run_key": child_key,
                "parent_run_key": state.get("current_parent_run_key") or "",
                "mutation": mutation,
                "score": (float(state.get("current_parent_score") or 0.0) + 0.1),
                "score_detail": {"dry_run": True},
                "status": "promotable",
                "completed_utc": now_utc(),
            }
            state["pending_candidates"] = [candidate]
            state["status"] = "waiting_for_promotion"
            append_event(cdir, "dry_run_candidate_created", candidate=candidate)
        else:
            before = {str(row.get("run_key") or "") for row in discover_completed_runs(root)}
            state["active_child_run_key"] = f"launching:{mutation['mutation_id']}"
            write_json(cdir / "state.json", state)
            append_event(cdir, "child_launch_started", mutation=mutation, runs_root=str(root))
            launch = launch_child(campaign_id=cid, mutation=mutation, runs_root=root)
            after_rows = discover_completed_runs(root)
            new_rows = [row for row in after_rows if str(row.get("run_key") or "") not in before]
            new_rows.sort(key=lambda row: float(row.get("score") or -999999.0), reverse=True)
            state["active_child_run_key"] = ""
            if not bool(launch.get("ok")) or not new_rows:
                failed = {
                    "run_key": f"failed:{mutation['mutation_id']}",
                    "parent_run_key": state.get("current_parent_run_key") or "",
                    "mutation": mutation,
                    "score": None,
                    "status": "failed",
                    "completed_utc": now_utc(),
                    "launch": launch,
                }
                rejected = list(state.get("rejected_candidates") or [])
                rejected.append(failed)
                state["rejected_candidates"] = rejected[-200:]
                state["consecutive_rejections"] = int(state.get("consecutive_rejections") or 0) + 1
                append_event(cdir, "child_launch_failed", candidate=failed)
            else:
                best_new = dict(new_rows[0])
                parent_score = state.get("current_parent_score")
                candidate = {
                    "run_key": best_new.get("run_key"),
                    "run_dir": best_new.get("run_dir"),
                    "parent_run_key": state.get("current_parent_run_key") or "",
                    "mutation": mutation,
                    "score": best_new.get("score"),
                    "score_detail": best_new.get("score_detail"),
                    "status": "promotable" if parent_score is None or float(best_new.get("score") or -999999.0) > float(parent_score or -999999.0) else "rejected",
                    "completed_utc": now_utc(),
                    "launch": launch,
                }
                if candidate["status"] == "promotable":
                    state["pending_candidates"] = [candidate]
                    state["status"] = "waiting_for_promotion"
                    append_event(cdir, "candidate_waiting_for_promotion", candidate=candidate)
                else:
                    rejected = list(state.get("rejected_candidates") or [])
                    rejected.append(candidate)
                    state["rejected_candidates"] = rejected[-200:]
                    state["consecutive_rejections"] = int(state.get("consecutive_rejections") or 0) + 1
                    append_event(cdir, "candidate_rejected_by_score", candidate=candidate)
    state["updated_utc"] = now_utc()
    write_json(cdir / "state.json", state)
    _write_lineage(cdir, state)
    _write_summary(cdir, state)
    return _public_campaign_payload(cdir, state)


def launch_child(*, campaign_id: str, mutation: Dict[str, Any], runs_root: Path) -> Dict[str, Any]:
    runs_root = Path(runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    runner = repo_root() / "_runner" / "run_registry_batch.py"
    batch_tag = f"{safe_campaign_id(campaign_id)}_{str(mutation.get('mutation_id') or 'mutation')}"
    cmd = [
        sys.executable,
        str(runner),
        "--registry",
        default_registry(),
        "--runs_root",
        str(runs_root),
        "--batch_tag",
        batch_tag,
    ]
    for knob in list(mutation.get("knobs") or []):
        text = str(knob or "").strip()
        if text:
            cmd.extend(["--knob", text])
    started = now_utc()
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "schema_version": "pressure_search_launch_v1",
        "ok": proc.returncode == 0,
        "started_utc": started,
        "finished_utc": now_utc(),
        "returncode": int(proc.returncode),
        "command": cmd,
        "stdout_tail": (proc.stdout or "")[-8000:],
    }


def _write_lineage(cdir: Path, state: Dict[str, Any]) -> None:
    nodes = []
    parent = str(state.get("current_parent_run_key") or "")
    if parent:
        nodes.append({"run_key": parent, "kind": "current_parent", "score": state.get("current_parent_score")})
    for key in ("pending_candidates", "rejected_candidates", "parked_candidates"):
        for row in list(state.get(key) or []):
            nodes.append({**dict(row), "kind": key.replace("_candidates", "")})
    write_json(
        cdir / "lineage.json",
        {
            "schema_version": LINEAGE_SCHEMA_VERSION,
            "utc": now_utc(),
            "campaign_id": state.get("campaign_id"),
            "nodes": nodes,
        },
    )


def _write_summary(cdir: Path, state: Dict[str, Any]) -> None:
    lines = [
        f"# Pressure Search {state.get('campaign_id')}",
        "",
        f"- status: {state.get('status')}",
        f"- current_parent: {state.get('current_parent_run_key') or '(none)'}",
        f"- current_parent_score: {state.get('current_parent_score')}",
        f"- active_child: {state.get('active_child_run_key') or '(none)'}",
        f"- pending_candidates: {len(list(state.get('pending_candidates') or []))}",
        f"- rejected_candidates: {len(list(state.get('rejected_candidates') or []))}",
        f"- updated_utc: {state.get('updated_utc')}",
    ]
    mutation = state.get("last_mutation") if isinstance(state.get("last_mutation"), dict) else {}
    if mutation:
        lines.extend(["", "## Last Mutation", "", str(mutation.get("summary") or mutation)])
    (cdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _public_campaign_payload(cdir: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **dict(state),
        "campaign_dir": str(cdir),
        "control": read_json(cdir / "control.json", default_control()),
        "summary_path": str(cdir / "summary.md"),
        "lineage_path": str(cdir / "lineage.json"),
        "events_path": str(cdir / "events.jsonl"),
    }


def run_loop(args: argparse.Namespace) -> int:
    deadline = time.time() + max(1, int(args.duration_days)) * 86400
    ensure_campaign(args.campaign_id, duration_days=int(args.duration_days), max_active_children=int(args.max_active_children))
    while True:
        payload = supervisor_step(
            args.campaign_id,
            runs_root=Path(args.runs_root) if str(args.runs_root).strip() else default_runs_root(),
            dry_run=bool(args.dry_run),
        )
        print(json.dumps({"campaign_id": args.campaign_id, "status": payload.get("status"), "updated_utc": payload.get("updated_utc")}, sort_keys=True), flush=True)
        if bool(args.once) or payload.get("status") in {"stopped", "failed"}:
            return 0
        if time.time() >= deadline:
            request_action(args.campaign_id, "stop", note="duration elapsed", requested_by="supervisor")
            return 0
        time.sleep(max(5, int(args.poll_seconds)))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Controlled Emergence pressure-point search supervisor")
    p.add_argument("--campaign_id", default=DEFAULT_CAMPAIGN_ID)
    p.add_argument("--runs_root", default=str(default_runs_root()))
    p.add_argument("--duration_days", type=int, default=28)
    p.add_argument("--max_active_children", type=int, default=1)
    p.add_argument("--profile", default="evolvability_band")
    p.add_argument("--poll_seconds", type=int, default=60)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--once", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
