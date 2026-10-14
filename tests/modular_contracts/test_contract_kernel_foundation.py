from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import jsonschema
import pytest

import emergence_contracts
from emergence_contracts import (
    ArtifactId,
    ComponentCompatibilityV1,
    ComponentDescriptorV1,
    ComponentId,
    ContentDigest,
    ContractError,
    ContractErrorCategory,
    ModuleId,
    RunId,
    SchemaVersion,
    canonical_json_bytes,
    canonical_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _descriptor() -> ComponentDescriptorV1:
    return ComponentDescriptorV1(
        component_id=ComponentId("world.patch.v1"),
        module_id=ModuleId("emergence_worldkit"),
        semantic_contract_version=SchemaVersion("world_contract_v1"),
        implementation_version=SchemaVersion("legacy_patch_v1"),
        capabilities=("observe", "step"),
        parameter_schema_ref="urn:eiel:world:patch-config-v1",
        state_schema_refs=("urn:eiel:world:patch-state-v1",),
        observation_schema_refs=("urn:eiel:world:patch-observation-v1",),
        action_schema_refs=("urn:eiel:world:patch-action-v1",),
        native_binding_ids=(ArtifactId("patch-native-binding-v1"),),
        compatibility=ComponentCompatibilityV1(
            semantic_contract_versions=(SchemaVersion("world_contract_v1"),),
            state_schema_versions=(SchemaVersion("patch_state_v1"),),
        ),
    )


def test_clean_import_is_inert_and_exposes_package_version() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import json, sys; import emergence_contracts as c; "
            "print(json.dumps({'version': c.__version__, "
            "'forbidden': sorted(name for name in sys.modules if "
            "name.split('.', 1)[0] in {'engine','tools','experiments',"
            "'experiments_shared','torch','numpy'})}))"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result == {"version": emergence_contracts.__version__, "forbidden": []}
    assert emergence_contracts.C001_API_VERSION == SchemaVersion("c001_v1")
    assert emergence_contracts.C010_API_VERSION == SchemaVersion("c010_v1")


@pytest.mark.parametrize("value", ["run-001", "legacy/value:exact", "ü-opaque"])
def test_opaque_ids_preserve_values_exactly(value: str) -> None:
    identity = RunId(value)
    assert identity.value == value
    assert str(identity) == value


def test_opaque_ids_are_immutable() -> None:
    identity = RunId("run-001")
    with pytest.raises(FrozenInstanceError):
        identity.value = "changed"  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        identity.runtime_owner = object()  # type: ignore[attr-defined]


@pytest.mark.parametrize("value", ["", " padded", "padded ", "bad\nvalue", 7])
def test_opaque_ids_fail_closed(value: object) -> None:
    with pytest.raises(ContractError) as error:
        RunId(value)  # type: ignore[arg-type]
    assert error.value.category is ContractErrorCategory.CONTRACT_INVALID
    assert error.value.owner == "emergence_contracts"


@pytest.mark.parametrize("value", ["", "v 1", "v/1", " v1", "v1\n"])
def test_schema_version_validation(value: str) -> None:
    with pytest.raises(ContractError):
        SchemaVersion(value)
    assert str(SchemaVersion("component_descriptor_v1")) == "component_descriptor_v1"
    assert str(SchemaVersion("1.0.0+native")) == "1.0.0+native"


def test_content_digest_validation_and_round_trip() -> None:
    digest = ContentDigest("sha256", "a" * 64)
    assert str(digest) == f"sha256:{'a' * 64}"
    assert ContentDigest.parse(str(digest)) == digest
    assert digest.to_dict() == {"algorithm": "sha256", "hex": "a" * 64}
    for invalid in (
        ("sha1", "a" * 40),
        ("sha256", "A" * 64),
        ("sha256", "a" * 63),
        ("sha256", "g" * 64),
    ):
        with pytest.raises(ContractError):
            ContentDigest(*invalid)


def test_canonical_json_is_sorted_compact_ascii_and_deterministic() -> None:
    first = {"z": [3, 2, 1], "é": {"b": False, "a": None}}
    second = {"é": {"a": None, "b": False}, "z": (3, 2, 1)}
    expected = b'{"z":[3,2,1],"\\u00e9":{"a":null,"b":false}}'
    assert canonical_json_bytes(first) == expected
    assert canonical_json_bytes(second) == expected
    assert canonical_sha256(first) == canonical_sha256(second)


@pytest.mark.parametrize(
    "value",
    [
        {"bad": {1, 2}},
        {1: "non-string-key"},
        {"bad": object()},
        {"bad": b"bytes"},
    ],
)
def test_canonical_json_rejects_unsupported_values(value: object) -> None:
    with pytest.raises(ContractError) as error:
        canonical_json_bytes(value)
    assert error.value.code == "CONTRACT_INVALID"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ContractError):
        canonical_json_bytes({"value": value})


def test_canonical_json_rejects_cycles() -> None:
    value: list[object] = []
    value.append(value)
    with pytest.raises(ContractError):
        canonical_json_bytes(value)


def test_typed_error_categories_are_machine_readable() -> None:
    assert {item.value for item in ContractErrorCategory} == {
        "CONTRACT_INVALID",
        "CONFIGURATION_INVALID",
        "CAPABILITY_UNSUPPORTED",
        "DETERMINISM_VIOLATION",
        "STATE_INTEGRITY_FAILURE",
        "CHECKPOINT_INCOMPATIBLE",
        "RUNTIME_EXECUTION_FAILURE",
        "RESOURCE_UNAVAILABLE",
        "EXPERIMENT_INCOMPLETE",
        "EVIDENCE_INVALID",
        "QUALIFICATION_BLOCKED",
    }
    error = ContractError(
        ContractErrorCategory.CAPABILITY_UNSUPPORTED,
        owner="emergence_runtime",
        message="capability is not available",
        details={"capability": "example"},
    )
    assert error.to_dict() == {
        "category": "CAPABILITY_UNSUPPORTED",
        "owner": "emergence_runtime",
        "message": "capability is not available",
        "details": {"capability": "example"},
    }


def test_component_descriptor_is_immutable_canonical_and_schema_valid() -> None:
    descriptor = _descriptor()
    with pytest.raises(FrozenInstanceError):
        descriptor.module_id = ModuleId("other")  # type: ignore[misc]
    assert descriptor.canonical_bytes() == canonical_json_bytes(descriptor.to_dict())
    assert descriptor.content_digest() == canonical_sha256(descriptor.to_dict())
    schema_path = (
        REPO_ROOT
        / "emergence_contracts"
        / "schemas"
        / "component_descriptor_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(descriptor.to_dict(), schema)
    assert descriptor.to_dict()["schema_version"] == "component_descriptor_v1"
    invalid = descriptor.to_dict()
    invalid["component_id"] = " padded"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)


def test_component_descriptor_has_no_implementation_references() -> None:
    descriptor = _descriptor()
    serialized = descriptor.to_dict()
    assert "implementation_object" not in serialized
    assert "filesystem_root" not in serialized
    assert all(
        type(value) in {str, list, dict, type(None)}
        for value in serialized.values()
    )


def test_descriptor_rejects_noncanonical_or_wrong_typed_values() -> None:
    with pytest.raises(ContractError):
        ComponentDescriptorV1(
            component_id=ComponentId("component"),
            module_id=ModuleId("module"),
            semantic_contract_version=SchemaVersion("contract_v1"),
            implementation_version=SchemaVersion("implementation_v1"),
            capabilities=("z", "a"),
        )
    with pytest.raises(ContractError):
        ComponentDescriptorV1(
            component_id=ComponentId("component"),
            module_id=ModuleId("module"),
            semantic_contract_version=SchemaVersion("contract_v1"),
            implementation_version=SchemaVersion("implementation_v1"),
            native_binding_ids=("not-an-artifact-id",),  # type: ignore[arg-type]
        )
