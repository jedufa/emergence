from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    assert isinstance(obj, dict)
    return obj


def test_golden_references_layout_and_schema() -> None:
    root = Path(__file__).resolve().parent
    cases = {
        "chemistry_single_tick": "golden_chemistry_single_tick",
        "structure_single_operator": "golden_structure_single_operator",
        "recombination_single_cross": "golden_recombination_single_cross",
    }

    for case_dir, case_id in cases.items():
        p = root / case_dir
        assert p.is_dir(), f"missing case directory: {p}"

        inp = _load_json(p / "reference_input.json")
        out = _load_json(p / "expected_output.json")
        delta = _load_json(p / "expected_rng_delta.json")

        assert inp.get("case_id") == case_id
        assert inp.get("schema_version") == "golden_reference_input_v1"

        assert out.get("case_id") == case_id
        assert out.get("schema_version") == "golden_reference_expected_output_v1"

        assert delta.get("case_id") == case_id
        assert delta.get("schema_version") == "golden_reference_expected_rng_delta_v1"
        assert isinstance(delta.get("expected_delta"), dict)
