"""experiments.projects.emergence.experiments.harnesses.generated.20k_autopilot__8ad6ad__harness

Auto-generated harness runner (Run Builder).

What it does
------------
- Executes the experiment registry via experiments.projects.emergence.experiments.harness.run_experiments.main
- Writes a lightweight report.json in the batch directory (best-effort)

Usage
-----
python -m experiments.projects.emergence.experiments.harnesses.generated.20k_autopilot__8ad6ad__harness -- <any passthrough args>
  (unmanaged, legacy)

python -m experiments.projects.emergence.tools.core.run_job --module experiments.projects.emergence.experiments.harnesses.generated.20k_autopilot__8ad6ad__harness --git_ref <40hex_or_tag> --job_id <job_id> --seed <seed> -- <any passthrough args>
  (managed: durable report_dir + stop sentinels)

Environment
-----------
EMERGENCE_OUTPUTS_DIR overrides the default runs_root.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import shlex
import shutil
from pathlib import Path


TEST_ID = '20k_autopilot'
CANONICAL_UID = '20k_autopilot'
REGISTRY_REL_PATH = '_spec/registry/v1/generated/20k_autopilot__8ad6ad__harness.yaml'
DEFAULT_RUNS_ROOT = './experiments/projects/emergence/_artifacts/runs'


def _project_root() -> Path:
    """Return Emergence project root (best-effort).

    This file is expected to live at:
      <repo>/experiments/projects/emergence/experiments/harnesses/generated/<name>__harness.py
    """

    return Path(__file__).resolve().parents[3]


from experiments.projects.emergence.experiments.harness.run_experiments import main as harness_main  # type: ignore  # noqa: E402


def _write_report(batch_dir: Path, *, runs_root: str, batch_tag: str) -> None:
    """Write report.json to the batch directory (best-effort).

    The goal is not to be fancy; it's to leave a breadcrumb trail.
    """

    try:
        summary_path = batch_dir / "summary.json"
        runs = vku._summary_runs(summary_path) if summary_path.exists() else []
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
                {"run_id": r.get("run_id"), "status": r.get("status"), "run_dir": r.get("run_dir")}
                for r in (runs or [])
            ],
        }
        (batch_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        return


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    argv = list(sys.argv[1:] if argv is None else argv)

    runs_root = os.environ.get("EMERGENCE_OUTPUTS_DIR") or os.environ.get("EXPERIMENT_OUTPUTS_DIR") or DEFAULT_RUNS_ROOT
    registry_path = str((_project_root() / REGISTRY_REL_PATH).resolve())
    batch_tag = TEST_ID

    rc = harness_main(["--registry", registry_path, "--runs_root", runs_root, "--batch_tag", batch_tag] + argv)

    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
