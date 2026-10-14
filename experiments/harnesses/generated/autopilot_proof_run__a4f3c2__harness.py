"""experiments.harnesses.generated.autopilot_proof_run__a4f3c2__harness

Generated-style harness for a short, aggressive pressure-autopilot proof run.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore


TEST_ID = "autopilot_proof_run"
CANONICAL_UID = "autopilot_proof_run"
REGISTRY_REL_PATH = "_spec/registry/v1/generated/autopilot_proof_run__a4f3c2__harness.yaml"
DEFAULT_RUNS_ROOT = "./experiments/projects/emergence/_artifacts/runs"


def _project_root() -> Path:
    """Return Emergence project root (best-effort)."""

    return Path(__file__).resolve().parents[3]


from experiments.projects.emergence.experiments.harness.run_experiments import main as harness_main  # type: ignore  # noqa: E402
from experiments.projects.emergence.experiments.runs import validation_knob_utils as vku  # noqa: E402


def _load_registry_identity(registry_path: str) -> tuple[str, int]:
    """Return registry name/version from YAML."""

    raw = Path(registry_path).read_text(encoding="utf-8")
    doc = yaml.safe_load(raw)
    if not isinstance(doc, dict):
        raise ValueError("Registry YAML must parse into a mapping.")
    return str(doc.get("name") or ""), int(doc.get("version") or 1)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file as a list of dict records."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def _summarize_autopilot(run_dir: Path) -> dict[str, Any]:
    """Summarize proof-run evidence from the autopilot decisions log."""

    decisions_path = run_dir / "autopilot" / "pressure_autopilot_decisions.jsonl"
    decisions = _read_jsonl(decisions_path)

    accepted = [d for d in decisions if str(d.get("ack_status", "")) == "ACCEPTED"]
    non_empty = [d for d in decisions if isinstance(d.get("updates"), list) and d.get("updates")]
    axes_seen = sorted(
        {
            str(d.get("axis"))
            for d in decisions
            if str(d.get("axis") or "") in {"scarcity", "cost", "shock", "all"}
        }
    )
    update_paths_seen = sorted(
        {
            str(u.get("path"))
            for d in decisions
            for u in (d.get("updates") or [])
            if isinstance(u, dict) and str(u.get("path") or "")
        }
    )
    generations = [
        int(d.get("generation"))
        for d in decisions
        if isinstance(d.get("generation"), int)
    ]
    tick_deltas = [
        generations[i] - generations[i - 1]
        for i in range(1, len(generations))
        if generations[i] >= generations[i - 1]
    ]
    evidence_ok = bool(decisions and accepted and non_empty)

    return {
        "decisions_path": str(decisions_path),
        "decisions_present": bool(decisions),
        "decision_count": int(len(decisions)),
        "accepted_count": int(len(accepted)),
        "non_empty_update_count": int(len(non_empty)),
        "axes_seen": axes_seen,
        "update_paths_seen": update_paths_seen,
        "first_generation": min(generations) if generations else None,
        "last_generation": max(generations) if generations else None,
        "tick_deltas_head": tick_deltas[:10],
        "evidence_ok": evidence_ok,
    }


def _write_report(batch_dir: Path, *, runs_root: str, batch_tag: str) -> None:
    """Write a minimal batch report for quick operator inspection."""

    try:
        runs = vku._summary_runs(batch_dir)
        report = {
            "report_api": 1,
            "utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "test_id": TEST_ID,
            "canonical_uid": CANONICAL_UID,
            "batch_tag": batch_tag,
            "runs_root": str(runs_root),
            "batch_dir": str(batch_dir),
            "registry": REGISTRY_REL_PATH,
            "total_runs": int(len(runs)) if isinstance(runs, list) else 0,
            "runs": [
                {
                    "run_id": r.get("run_id"),
                    "status": r.get("status"),
                    "run_dir": r.get("run_dir"),
                    "autopilot": _summarize_autopilot(Path(str(r.get("run_dir", "")))),
                }
                for r in (runs or [])
            ],
        }
        (batch_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        return


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    argv = list(sys.argv[1:] if argv is None else argv)

    runs_root = os.environ.get("EMERGENCE_OUTPUTS_DIR") or os.environ.get("EXPERIMENT_OUTPUTS_DIR") or DEFAULT_RUNS_ROOT
    registry_path = str((_project_root() / REGISTRY_REL_PATH).resolve())
    batch_tag = TEST_ID

    rc = harness_main(["--registry", registry_path, "--runs_root", runs_root, "--batch_tag", batch_tag] + argv)

    try:
        reg_name, reg_version = _load_registry_identity(registry_path)
        batch_dir = vku._find_batch_dir(vku._resolve_runs_root(runs_root), reg_name, reg_version, batch_tag)
        if batch_dir:
            _write_report(batch_dir, runs_root=runs_root, batch_tag=batch_tag)
    except Exception:
        pass

    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
