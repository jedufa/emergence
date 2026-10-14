from __future__ import annotations

from pathlib import Path

import pytest

from tools.performance import build_flat_resident_reporting_contract as contract_builder


def test_contract_inventory_contains_exactly_66_unique_complete_columns() -> None:
    contract, schema = contract_builder._build_payloads()

    fairness_columns = contract["fairness_columns"]
    contract_columns = [column["column"] for column in contract["columns"]]

    assert len(fairness_columns) == len(set(fairness_columns)) == 66
    assert contract["column_count"] == schema["fairness_column_count"] == 66
    assert contract_columns == fairness_columns == list(contract_builder.FAIRNESS_COLUMNS)
    assert all(column["primitive_keys"] for column in contract["columns"])


def test_every_fairness_primitive_native_mapping_is_discoverable() -> None:
    contract, schema = contract_builder._build_payloads()

    assert schema["field_count"] == len(schema["fields"])
    assert len({field["key"] for field in schema["fields"]}) == schema["field_count"]
    for field in schema["fields"]:
        mapping = field["native_mapping"]
        assert mapping["status"] == "discoverable", field["key"]
        assert mapping["candidate_keys"], field["key"]
        assert set(mapping["discoveries"]) == set(mapping["candidate_keys"])
        assert all(mapping["discoveries"].values()), field["key"]

    assert contract["validation"]["all_primitive_keys_have_discoverable_native_mappings"] is True
    assert "all_primitive_keys_have_native_mapping_or_placeholder" not in contract["validation"]


def test_contract_contains_no_unresolved_native_primitive_keys() -> None:
    contract, _ = contract_builder._build_payloads()

    for column in contract["columns"]:
        assert column["native_mapping"]["status"] == "discoverable", column["column"]
        assert column["native_mapping"]["unmapped_primitive_keys"] == [], column["column"]


def test_maturity_age_maps_to_shared_developmental_config_metadata() -> None:
    _, schema = contract_builder._build_payloads()
    maturity_field = next(field for field in schema["fields"] if field["key"] == "maturity_age")
    mapping = maturity_field["native_mapping"]

    assert maturity_field["reference_adapter"]["expression"] == "float(phenotype.maturity_age)"
    assert mapping["kind"] == "shared_config_metadata"
    assert mapping["candidate_keys"] == ["developmental_maturity_age"]
    assert "sim_cfg.developmental_maturity_age" in mapping["note"]
    assert "metadata" in mapping["note"]

    resident_source = Path("engine/perf/cxxbackbone10_resident_state_owner.py")
    discoveries = mapping["discoveries"]["developmental_maturity_age"]
    resident_discovery = next(
        discovery for discovery in discoveries if discovery["source"] == str(resident_source)
    )
    source_line = (contract_builder.PROJECT_ROOT / resident_source).read_text(encoding="utf-8").splitlines()[
        resident_discovery["line"] - 1
    ]
    assert '"developmental_maturity_age"' in source_line
    assert 'getattr(sim_cfg, "developmental_maturity_age"' in source_line


@pytest.mark.parametrize(
    "hint",
    [
        contract_builder.NativeHint("placeholder", (), "unresolved test hint"),
        contract_builder.NativeHint(
            "shared_config_metadata",
            ("not_a_resident_metadata_key",),
            "unmapped test hint",
        ),
        contract_builder.NativeHint(
            "shared_config_metadata",
            ("developmental_maturity_age", "not_a_resident_metadata_key"),
            "partial test hint",
        ),
    ],
)
def test_generation_fails_closed_for_any_unresolved_fairness_primitive(
    monkeypatch: pytest.MonkeyPatch,
    hint: contract_builder.NativeHint,
) -> None:
    monkeypatch.setitem(contract_builder.NATIVE_FIELD_HINTS, "maturity_age", hint)

    with pytest.raises(
        contract_builder.ContractError,
        match=r"native mapping coverage failure: unresolved=\['maturity_age'\]",
    ):
        contract_builder._build_payloads()
