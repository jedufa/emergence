from __future__ import annotations

import json

from tools.scripts import eiel_b48_wdl_jungle_eed_hapb_h1_verdict_v1 as subject


def test_freeze_selects_only_the_required_worst_case_hot_prefix_policy(tmp_path) -> None:
    receipt = {
        "schema_version": "eiel_b48_wdl_jungle_eed_hapb_h1_canary_v1",
        "frozen_gate": {"maximum_g200_seconds": 10.0},
        "receipt_sha256": "source",
        "fully_pinned_admitted": False,
        "variants": [
            {"representation": "p256_hot_prefix", "active_rows_per_cell": 512, "tile_rows": 512, "buffer_count": 1, "canonical_host_storage": "pageable", "host_pressure_passed": True, "projected_g200_seconds": 9.0},
        ],
    }
    path = tmp_path / "h1.json"
    output = tmp_path / "verdict.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")

    result = subject.freeze_h1_verdict_v1(path, output=output)

    assert result["verdict"] == "PASS_START_H2"
    assert result["selected_representation"]["tile_rows"] == 512
