from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_eiel_authoritative_state_production_identity_v1 import _identity
from tools.scripts import eiel_authoritative_state_admission_performance_v1 as subject


def _manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "artifact_identity": {
                    "runtime_resource_receipt": {
                        "authoritative_state_identity_v1": _identity()
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_receipt_measures_only_pre_dispatch_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    (tmp_path / "engine").mkdir()
    clock = iter(value for _ in range(10) for value in (0, 100))
    monkeypatch.setattr(subject.time, "perf_counter_ns", lambda: next(clock))

    receipt = subject.build_receipt(
        manifest_path=manifest,
        source_root=tmp_path,
        iterations=10,
        batch_cells=2,
        generations=3,
        ticks_per_generation=5,
    )

    assert receipt["admission_validation_wall_ns"] == {
        "minimum": 100,
        "median": 100,
        "p95": 100,
        "maximum": 100,
    }
    assert receipt["execution_envelope"]["physical_cell_ticks"] == 30
    assert receipt["amortized_admission_ns_per_physical_cell_tick"] == 100 / 30
    assert receipt["native_tick_admission_call_site_count"] == 0
    assert receipt["per_tick_validation_added"] is False
    assert receipt["incremental_post_dispatch_validation_overhead_ns_per_tick"] == 0


def test_native_admission_reference_fails_closed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    source = tmp_path / "engine" / "native"
    source.mkdir(parents=True)
    (source / "tick.cu").write_text(
        "void tick() { validate_admission_identity_v1(); }\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="native tick sources"):
        subject.build_receipt(
            manifest_path=manifest,
            source_root=tmp_path,
            iterations=10,
            batch_cells=1,
            generations=1,
            ticks_per_generation=1,
        )
