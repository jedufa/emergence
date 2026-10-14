from __future__ import annotations

import json
from pathlib import Path

from tools.scripts.eiel_option_d_m27_env_policy_pcg64_readiness_audit import (
    DECISION_WITHHELD,
    REQUIRED_SOURCE_PATTERNS,
    decide,
)


def _write_required_sources(root: Path) -> None:
    for relative, patterns in REQUIRED_SOURCE_PATTERNS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(patterns) + "\n", encoding="utf-8")


def test_m27_identifies_env_policy_operations_and_withholds_promotion(
    tmp_path: Path,
) -> None:
    _write_required_sources(tmp_path)
    m15 = tmp_path / "m15.json"
    m15.write_text(
        json.dumps(
            {
                "active_physical_streams": [
                    {"physical_stream_id": "numpy.environment_policy"}
                ]
            }
        ),
        encoding="utf-8",
    )
    m24 = tmp_path / "m24.json"
    m24.write_text(
        json.dumps({"missing_required_streams": ["numpy.environment_policy"]}),
        encoding="utf-8",
    )

    result = decide(repo_root=tmp_path, m15_receipt=m15, m24_receipt=m24)

    assert result["ok"] is True
    assert result["decision"] == DECISION_WITHHELD
    assert result["promotion_authorized"] is False
    assert result["receipt_implementation_authorized"] is True
    assert [row["operator_id"] for row in result["discovered_operations"]] == [
        "environment.food_spawn.uniform_or_bounded",
        "policy.action_gumbel.float32",
    ]
    assert "env_policy_operation_receipts_absent" in result["blockers"]
    assert len(result["receipt_sha256"]) == 64


def test_m27_fails_when_source_pattern_missing(tmp_path: Path) -> None:
    _write_required_sources(tmp_path)
    kernel = (
        tmp_path
        / "engine/perf/cuda_architecture_b_flattened_controller_v1_kernels.cu"
    )
    kernel.write_text("spawn_flat_food_exact\n", encoding="utf-8")
    m15 = tmp_path / "m15.json"
    m15.write_text(
        json.dumps(
            {
                "active_physical_streams": [
                    {"physical_stream_id": "numpy.environment_policy"}
                ]
            }
        ),
        encoding="utf-8",
    )
    m24 = tmp_path / "m24.json"
    m24.write_text(
        json.dumps({"missing_required_streams": ["numpy.environment_policy"]}),
        encoding="utf-8",
    )

    result = decide(repo_root=tmp_path, m15_receipt=m15, m24_receipt=m24)

    assert result["ok"] is False
    assert "required_source_patterns_missing" in result["blockers"]
    assert result["missing_source_patterns"]
