from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from eiel_composition.science_instrument_cli import main
from eiel_composition.science_instrument_demo_v1 import (
    build_science_functional_demonstration_v1,
)
from emergence_contracts import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]


def test_functional_demonstration_is_deterministic_and_truthfully_fail_closed() -> None:
    first = build_science_functional_demonstration_v1()
    second = build_science_functional_demonstration_v1()

    assert first == second
    assert first["status"] == "TECHNICAL_DEMONSTRATION_PASS"
    assert first["primary_outcome"] == "OUTSIDE_QUALIFIED_ENVELOPE"
    assert first["semantic_cell_count"] == first["world_count"] == 12
    assert first["record_count"] == first["partition_count"] == 24
    assert all(first["invariants"].values())
    assert first["gpu_used"] is False
    assert first["qualification_started"] is False
    assert first["qualification_authority_effect"] is False
    assert (
        first["receipt_sha256"]
        == canonical_sha256(
            {key: value for key, value in first.items() if key != "receipt_sha256"}
        ).to_dict()
    )


def test_functional_demonstration_matches_portable_schema() -> None:
    result = build_science_functional_demonstration_v1()
    schema = json.loads(
        (
            ROOT
            / "_spec/science_instrument/v1/science_functional_demonstration.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(result, schema)


def test_science_demo_cli_prints_and_can_retain_the_same_receipt(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "demo.json"
    assert main(["demo", "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    retained = json.loads(output.read_text(encoding="utf-8"))

    assert printed == retained == build_science_functional_demonstration_v1()
