from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/scripts/eiel_authoritative_state_abi_v1.py"
CONTRACT = ROOT / "_spec/authoritative_state/v1/abi_variants.json"


def _module():
    spec = importlib.util.spec_from_file_location("authoritative_state_abi_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_contract_is_deterministic_and_current() -> None:
    module = _module()
    observed = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert observed == module.build_contract()
    assert module.validate_contract(observed) == observed
    without_hash = dict(observed)
    digest = without_hash.pop("contract_sha256")
    assert digest == hashlib.sha256(
        module.canonical_json_bytes(without_hash)
    ).hexdigest()


def test_all_current_supported_variants_have_exact_orders() -> None:
    module = _module()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    variants = {item["variant_id"]: item for item in contract["variants"]}
    assert {key: value["count"] for key, value in variants.items()} == {
        "ordinary_birth_disabled_0": 0,
        "ordinary_birth_legacy_113": 113,
        "ordinary_birth_diploid_122": 122,
        "ordinary_birth_genome_v2_139": 139,
            "turnover_disabled_0": 0,
            "turnover_mutable_state_122": 122,
            "turnover_output_disabled_0": 0,
            "turnover_output_16": 16,
        "full_tick_state_29": 29,
        "event_state_31": 31,
        "ordinary_birth_output_disabled_0": 0,
        "ordinary_birth_output_63": 63,
        "ordinary_birth_scratch_disabled_0": 0,
        "ordinary_birth_scratch_1": 1,
        "patch_p2_state_disabled_0": 0,
        "patch_p2_state_context_7": 7,
        "patch_p2_state_full_10": 10,
        "direct_resident_outer_172": 172,
    }
    for variant in variants.values():
        names = [entry["name"] for entry in variant["entries"]]
        assert [entry["index"] for entry in variant["entries"]] == list(
            range(len(names))
        )
        assert variant["order_sha256"] == module.order_sha256(names)
        assert module.validate_order(
            contract,
            family=variant["abi_family"],
            names=names,
        ) == variant["variant_id"]
    turnover = variants["turnover_mutable_state_122"]["entries"]
    assert all("cpp_enum_name" in entry for entry in turnover)


def test_same_count_different_order_fails_closed() -> None:
    module = _module()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    legacy = next(
        item
        for item in contract["variants"]
        if item["variant_id"] == "ordinary_birth_legacy_113"
    )
    names = [entry["name"] for entry in legacy["entries"]]
    names[0], names[1] = names[1], names[0]
    with pytest.raises(module.AbiContractError, match="count 113 is known but order hash"):
        module.validate_order(
            contract,
            family="ordinary_birth_state",
            names=names,
        )


def test_tampered_contract_and_unknown_count_fail_closed() -> None:
    module = _module()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    tampered = json.loads(json.dumps(contract))
    nonempty = next(item for item in tampered["variants"] if item["entries"])
    nonempty["entries"][0]["name"] = "wrong"
    with pytest.raises(module.AbiContractError, match="differs"):
        module.validate_contract(tampered)
    with pytest.raises(module.AbiContractError, match="unsupported count"):
        module.validate_order(
            contract,
            family="ordinary_birth_state",
            names=["only_one"],
        )


def test_all_retained_families_are_supported_with_exact_metadata() -> None:
    module = _module()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    unsupported = {item["abi_family"] for item in contract["unsupported_families"]}
    assert unsupported == set()
    full_tick_names = [
        entry["name"]
        for entry in next(
            item
            for item in contract["variants"]
            if item["variant_id"] == "full_tick_state_29"
        )["entries"]
    ]
    assert module.validate_sequence(
        contract,
        family="full_tick_state",
        names=full_tick_names,
    ) == "full_tick_state_29"
    assert module.validate_sequence(
        contract,
        family="ordinary_birth_scratch",
        names=["lifecycle_scratch_storage"],
    ) == "ordinary_birth_scratch_1"


def test_every_nonempty_entry_has_required_metadata_keys() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    required = {
        "semantic_field_id",
        "physical_type",
        "dtype",
        "signedness",
        "rank",
        "shape_formula",
        "valid_row_count_formula",
        "capacity_formula",
        "mutability",
        "semantic_owner",
        "optionality",
        "checkpoint_participation",
        "chronicle_participation",
        "profile_availability",
        "metadata_status",
    }
    for variant in contract["variants"]:
        for entry in variant["entries"]:
            assert required <= entry.keys()


def test_all_source_anchors_are_hash_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for source in contract["sources"]:
        path = ROOT / source["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
