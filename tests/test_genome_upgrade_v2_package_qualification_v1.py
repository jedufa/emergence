from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.evolution.genome_upgrade_v2_package_v1 import (
    PRE_ROUTE_PACKAGE_OBLIGATIONS,
)
from tools.scripts.genome_upgrade_v2_package_qualification_v1 import (
    _evidence_rows,
)


def _bindings(tmp_path: Path, *, artifact_sha256: str) -> list[str]:
    static = {
        "genome_v2_default_off_exact",
        "genome_v2_neutral_profile_exact",
        "genome_v2_reference_native_cuda_expression_parity",
        "genome_v2_inheritance_mutation_parity",
    }
    bindings: list[str] = []
    for obligation in PRE_ROUTE_PACKAGE_OBLIGATIONS:
        path = tmp_path / f"{obligation}.json"
        payload: dict[str, object] = {"ok": True}
        if obligation not in static:
            payload["artifact_binding"] = {
                "artifact_sha256": artifact_sha256,
            }
        path.write_text(json.dumps(payload), encoding="utf-8")
        bindings.append(f"{obligation}={path}")
    return bindings


def test_artifact_bound_evidence_requires_exact_loaded_artifact(
    tmp_path: Path,
) -> None:
    expected = "a" * 64
    bindings = _bindings(tmp_path, artifact_sha256=expected)
    rows = _evidence_rows(bindings, artifact_sha256=expected)
    assert set(rows) == set(PRE_ROUTE_PACKAGE_OBLIGATIONS)

    dynamic = next(
        binding
        for binding in bindings
        if binding.startswith("genome_v2_cancellation=")
    )
    path = Path(dynamic.partition("=")[2])
    path.write_text(
        json.dumps({"artifact_binding": {"artifact_sha256": "b" * 64}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not declare the exact package artifact"):
        _evidence_rows(bindings, artifact_sha256=expected)
