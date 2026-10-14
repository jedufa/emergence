"""Compatibility wrapper over shared manifest job-spec helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from experiments_shared.job_manifests.job_spec import build_job_spec, render_cli_args


def _render_cli_args(args_dict: Dict[str, Any]) -> List[str]:
    return render_cli_args(args_dict)


def _build_job_spec(*, job_id: str, git_ref: str, module: str, args_dict: Dict[str, Any], seed: int) -> Dict[str, Any]:
    return build_job_spec(job_id=job_id, git_ref=git_ref, module=module, args_dict=args_dict, seed=seed)

