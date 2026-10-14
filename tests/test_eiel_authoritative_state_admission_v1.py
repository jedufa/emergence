from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from engine.authoritative_state.admission import (
    SchemaAdmissionError,
    canonical_sha256,
    resolve_migration,
    validate_compatibility_matrix,
    validate_contract_identity,
    validate_field_manifest,
    validate_migration_rules,
)
from engine.authoritative_state.schema import (
    AuthoritativeStateSchemaError,
    _validate_migration_families,
)


SPEC = Path(__file__).parents[1] / "_spec" / "authoritative_state" / "v1"


def _load(name: str) -> dict:
    return json.loads((SPEC / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "name",
    [
        "checkpoint_schema.json",
        "chronicle_schema.json",
        "provider_projection_schema.json",
        "analysis_projection_schema.json",
        "compatibility_matrix.json",
        "migration_rules.json",
        "migration_families.json",
    ],
)
def test_canonical_component_contract_hashes_are_valid(name: str) -> None:
    document = _load(name)
    assert validate_contract_identity(document) == document["contract_hash"]


def test_canonical_hash_rejects_nonfinite_values() -> None:
    with pytest.raises(SchemaAdmissionError, match="canonical JSON"):
        canonical_sha256({"bad": float("nan")})


def test_field_manifest_rejects_order_dtype_and_shape_drift() -> None:
    fields = _load("checkpoint_schema.json")["fields"]
    for mutation, message in (
        (lambda rows: rows.reverse(), "order"),
        (lambda rows: rows[0].__setitem__("dtype", "int32"), "dtype"),
        (lambda rows: rows[0].__setitem__("shape", [1]), "shape"),
    ):
        observed = deepcopy(fields)
        mutation(observed)
        with pytest.raises(SchemaAdmissionError, match=message):
            validate_field_manifest(fields, observed)


def test_compatibility_matrix_requires_one_exact_hash_tuple() -> None:
    matrix = _load("compatibility_matrix.json")
    observed = matrix["compatible_sets"][0]["bindings"]
    assert (
        validate_compatibility_matrix(matrix, observed=observed)["compatibility_id"]
        == "eiel_authoritative_state_components_v1"
    )
    bad = dict(observed)
    bad["checkpoint"] = "f" * 64
    with pytest.raises(SchemaAdmissionError, match="matches=0"):
        validate_compatibility_matrix(matrix, observed=bad)
    extra = dict(observed, binary="a" * 64)
    with pytest.raises(SchemaAdmissionError, match="matches=0"):
        validate_compatibility_matrix(matrix, observed=extra)


def test_only_explicit_deterministic_migration_is_admitted() -> None:
    migrations = _load("migration_rules.json")
    validate_migration_rules(migrations)
    digest = _load("checkpoint_schema.json")["contract_hash"]
    rule = resolve_migration(
        migrations,
        component="checkpoint",
        from_contract_hash=digest,
        to_contract_hash=digest,
    )
    assert rule["category"] == "identity"
    with pytest.raises(SchemaAdmissionError, match="found 0"):
        resolve_migration(
            migrations,
            component="checkpoint",
            from_contract_hash="a" * 64,
            to_contract_hash=digest,
        )


def test_migration_rule_requires_deterministic_category_payload() -> None:
    migrations = _load("migration_rules.json")
    bad = deepcopy(migrations)
    bad["rules"][0]["category"] = "add_required_with_constant"
    bad["rules"][0]["details"] = {"field_id": "checkpoint.generation.v1"}
    bad.pop("contract_hash")
    bad["contract_hash"] = canonical_sha256(bad)
    with pytest.raises(SchemaAdmissionError, match="deterministic value"):
        validate_migration_rules(bad)


def test_migration_family_matrix_classifies_current_and_legacy_surfaces() -> None:
    families = _load("migration_families.json")
    _validate_migration_families(families)
    by_id = {row["family_id"]: row for row in families["families"]}

    assert by_id["cross_path_checkpoint_schema_v1_current"]["category"] == (
        "compatible_no_op"
    )
    assert by_id["cross_path_checkpoint_set_v1_pre_authority"][
        "category"
    ] == "fresh_campaign"
    assert by_id["pre_schema_v1_cuda_binary"]["category"] == "unsupported"
    assert all(
        row["admission"] != "allowed_only_after_exact_identity_validation"
        for row in families["families"]
        if row["category"] in {"fresh_campaign", "unsupported"}
    )


def test_migration_family_matrix_rejects_contract_or_category_drift() -> None:
    families = _load("migration_families.json")
    bad_hash = deepcopy(families)
    bad_hash["families"][0]["operation"] = "silent_rewrite"
    with pytest.raises(
        AuthoritativeStateSchemaError,
        match="contract hash differs",
    ):
        _validate_migration_families(bad_hash)

    bad_category = deepcopy(families)
    bad_category["families"][0]["category"] = "best_effort"
    bad_category.pop("contract_hash")
    bad_category["contract_hash"] = canonical_sha256(bad_category)
    with pytest.raises(
        AuthoritativeStateSchemaError,
        match="category is invalid",
    ):
        _validate_migration_families(bad_category)
