from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tools.scripts import eiel_native_memory_first_difference_forensics as forensics


PROJECT = Path(__file__).resolve().parents[2]


def test_closeout_payloads_verify_and_self_checksum_is_explicit(tmp_path: Path) -> None:
    archive = (
        PROJECT / "evidences/90-diagnostics-and-closures/"
        "270-eiel-native-stable-memory-causal-closeout-20260721.evidence.zip"
    )

    forensics._audit_closeout_zip(archive, tmp_path)

    audit = forensics._read_json(tmp_path / "evidence_checksum_audit.json")
    assert audit["payload_entries"] == 22
    assert audit["payload_entries_verified"] == 22
    assert audit["payloads_all_verified"] is True
    assert audit["manifest_self_entry_valid"] is False


def test_historical_effects_recompute_from_archived_rows(tmp_path: Path) -> None:
    result = forensics._recompute_historical(PROJECT, tmp_path)

    assert result["decision"] == (
        "HISTORICAL-B/POSITIVE-WITH-MATERIALLY-DIFFERENT-SEMANTICS"
    )
    m1 = result["campaigns"]["m1"]
    m1r = result["campaigns"]["m1r"]
    assert m1["completed_runs"] == 160
    assert m1r["completed_runs"] == 160
    assert len(m1["supportive_rows"]) == 4
    assert len(m1r["supportive_rows"]) == 4
    assert m1["effects"]["init_legacy"]["interactions"]["tuned_jungle"][
        "alive"
    ] == pytest.approx(14.1)
    assert m1r["effects"]["init_legacy"]["interactions"]["tuned_jungle"][
        "mean_offspring_total"
    ] == pytest.approx(0.37039786496254534)


def test_native_reanalysis_preserves_all_seed_pairs(tmp_path: Path) -> None:
    native_root = Path(
        "/tmp/emergence-test/runtime/evidence_raw/eiel_native_stable_memory_science_20260721"
    )
    if not native_root.is_dir():
        pytest.skip("native forensic source evidence is unavailable")

    result = forensics._native_reanalysis(native_root, tmp_path)

    assert result["decision"] == "NATIVE-NEGATIVE-WDL-SPECIFICITY"
    with (tmp_path / "native_memory_seed_level_forensics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
    assert {int(row["seed"]) for row in rows} == set(forensics.SEEDS_NATIVE)


def test_first_material_difference_is_before_tick_zero(tmp_path: Path) -> None:
    rows = forensics._semantic_diff(tmp_path)

    founder_row = next(row for row in rows if row["field"] == "normal capacity")
    assert founder_row["category"] == "memory_arm"
    assert "every founder forced GRU" in founder_row["difference"]
    text = (tmp_path / "historical_native_first_material_difference.md").read_text(
        encoding="utf-8"
    )
    assert "before the first tick" in text
