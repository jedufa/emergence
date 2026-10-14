"""CRCA-G9 fail-closed optional-experiment and exception dispositions."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = (
    REPO_ROOT
    / "tools/architecture/baselines/eiel_modular_legacy_imports_v1.json"
)

EXPECTED_DISPOSITIONS = {
    "DEFAULT_OFF_NOT_STARTED_DEMONSTRATED_EXISTING_EXPERIMENTAL_CONSUMER": 6,
    "DEFAULT_OFF_NOT_STARTED_NO_DEMONSTRATED_NEW_EXTRACTION_CONSUMER": 1,
    "DEFAULT_OFF_NOT_STARTED_RETAINED_OPTION_D_ONLY": 2,
    "G7_ACCEPTED_CONTRACT_MIGRATION_EDGE_NOT_OPTIONAL_EXPERIMENT": 2,
    "G8_SHADOW_ONLY_NO_AUTHORITY_OR_FURTHER_EXTRACTION": 2,
    "NOT_SELECTED_IN_SINGLE_G7_SLICE_PRESERVED_FOR_FUTURE_OWNER": 8,
    "PRESERVED_UNCHANGED_COMPATIBILITY_EDGE": 6,
}


def _baseline() -> dict[str, object]:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_every_crca_expiry_has_an_explicit_g9_closeout_disposition() -> None:
    baseline = _baseline()
    closeout = baseline["crca_closeout"]
    assert closeout == {
        "ceiling_changes": False,
        "closed_gate": "CRCA-G9",
        "disposition_only": True,
        "programme_id": "CRCA-END-TO-END-20260823",
        "superseded_exception_records": 27,
    }
    exceptions = baseline["temporary_exceptions"]
    assert isinstance(exceptions, list)
    assert not any(
        item["expires_after_slice"].startswith("CRCA-") for item in exceptions
    )
    disposition_counts = Counter(
        item["crca_closeout_disposition"]
        for item in exceptions
        if item.get("crca_closeout_gate") == "CRCA-G9"
    )
    assert disposition_counts == EXPECTED_DISPOSITIONS


def test_rngv2_promotion_has_consumers_but_remains_default_off() -> None:
    baseline = _baseline()
    exceptions = baseline["temporary_exceptions"]
    rngv2 = [
        item
        for item in exceptions
        if item.get("crca_closeout_disposition")
        == "DEFAULT_OFF_NOT_STARTED_DEMONSTRATED_EXISTING_EXPERIMENTAL_CONSUMER"
    ]
    assert {(item["source"], item["target"]) for item in rngv2} == {
        ("campaigns", "hapb_rngv2"),
        ("hapb_rngv2", "tensor_world"),
        ("hapb_rngv2", "tw_rngv2"),
        ("perf", "hapb_rngv2"),
        ("perf", "tw_rngv2"),
        ("tensor_world", "tw_rngv2"),
    }
    assert all("future separately authorized" in item["expires_after_slice"] for item in rngv2)


def test_physical_and_option_d_experiments_remain_unstarted() -> None:
    baseline = _baseline()
    exceptions = baseline["temporary_exceptions"]
    physical = [
        item
        for item in exceptions
        if item.get("crca_closeout_disposition")
        == "DEFAULT_OFF_NOT_STARTED_NO_DEMONSTRATED_NEW_EXTRACTION_CONSUMER"
    ]
    assert [(item["source"], item["target"]) for item in physical] == [
        ("tensor_world", "agents")
    ]
    option_d = [
        item
        for item in exceptions
        if item.get("crca_closeout_disposition")
        == "DEFAULT_OFF_NOT_STARTED_RETAINED_OPTION_D_ONLY"
    ]
    assert {(item["source"], item["target"]) for item in option_d} == {
        ("option_d", "perf"),
        ("option_d", "tensor_world"),
    }
