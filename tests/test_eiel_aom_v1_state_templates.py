from __future__ import annotations

import copy
import json
import re
import zipfile
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_SECRET_KEYS = {
    "secret",
    "secrets",
    "token",
    "tokens",
    "password",
    "credential",
    "credentials",
    "private_key",
    "cookie",
    "cookies",
}


def _load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _validator(schema_path: str) -> Draft202012Validator:
    schema = _load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _front_matter(path: str) -> dict:
    text = (ROOT / path).read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, front_matter, _ = text.split("---", 2)
    data = yaml.safe_load(front_matter)
    assert isinstance(data, dict)
    return data


def _assert_valid(schema_path: str, value: dict) -> None:
    errors = sorted(_validator(schema_path).iter_errors(value), key=str)
    assert errors == []


def _assert_invalid(schema_path: str, value: dict, expected: str) -> None:
    errors = list(_validator(schema_path).iter_errors(value))
    assert errors
    text = " ".join([*map(str, errors[0].absolute_path), errors[0].message])
    assert expected in text


def _iter_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _iter_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_keys(child)


def _assert_no_path_escape(value: dict) -> None:
    for key in _iter_keys(value):
        assert ".." not in key


def _assert_no_secret_fields(value: dict) -> None:
    keys = {key.lower() for key in _iter_keys(value)}
    assert keys.isdisjoint(FORBIDDEN_SECRET_KEYS)


def _validate_prompt_receipt_semantics(receipt: dict, *, repo_root: Path) -> None:
    if receipt["supersedes_receipt"] == receipt["receipt_id"]:
        raise ValueError("receipt cannot supersede itself")
    if receipt["status"] == "active" and receipt["supersedes_receipt"]:
        raise ValueError("superseded receipt cannot be active")
    if receipt["status"] == "active":
        package = repo_root / receipt["work_package_path"]
        if not package.exists():
            raise ValueError("active receipt references missing work package")
        actual = _sha256(package)
        if receipt["work_package_sha256"] != actual:
            raise ValueError("active receipt work package hash mismatch")


def _validate_active_goal_semantics(goal: dict, *, repo_root: Path) -> None:
    for key in ("approved_work_package", "activation_prompt_receipt"):
        path = repo_root / goal[key]
        if not path.exists():
            raise ValueError(f"active goal references missing {key}")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_json_templates_parse_and_validate() -> None:
    cases = [
        (
            "bridge/templates/latest_status_v1.schema.json",
            "bridge/templates/latest_status_v1.json",
        ),
        (
            "bridge/templates/latest_status_v1.schema.json",
            "bridge/latest_status.json",
        ),
        (
            "bridge/templates/prompt_receipt_v1.schema.json",
            "bridge/templates/prompt_receipt_v1.json",
        ),
        (
            "work_packages/templates/active_goal_metadata_v1.schema.json",
            "work_packages/templates/active_goal_metadata_v1.json",
        ),
    ]
    for schema_path, example_path in cases:
        _assert_valid(schema_path, _load_json(example_path))


def test_work_package_front_matter_validates_and_has_no_transient_state() -> None:
    metadata = _front_matter("work_packages/templates/work_package_v1.md")
    _assert_valid("work_packages/templates/work_package_metadata_v1.schema.json", metadata)

    immutable_contract_fields = {
        "objective",
        "scope",
        "allowed_work",
        "forbidden_work",
        "hard_stop_conditions",
        "completion_criteria",
        "required_tests",
        "required_reports",
        "evidence_paths",
    }
    transient_fields = {
        "execution_status",
        "current_commit",
        "latest_commit",
        "commands_run",
        "tests_run",
        "retry_count",
    }
    assert immutable_contract_fields.issubset(metadata)
    assert transient_fields.isdisjoint(metadata)


def test_status_enums_reject_unknown_values() -> None:
    package = _front_matter("work_packages/templates/work_package_v1.md")
    package["status"] = "running"
    _assert_invalid(
        "work_packages/templates/work_package_metadata_v1.schema.json",
        package,
        "running",
    )

    status = _load_json("bridge/latest_status.json")
    status["execution_status"] = "approved"
    _assert_invalid(
        "bridge/templates/latest_status_v1.schema.json",
        status,
        "approved",
    )


def test_sha256_fields_require_lowercase_sha256() -> None:
    receipt = _load_json("bridge/templates/prompt_receipt_v1.json")
    receipt["source_sha256"] = "A" * 64
    _assert_invalid(
        "bridge/templates/prompt_receipt_v1.schema.json",
        receipt,
        "source_sha256",
    )


def test_prompt_receipt_packaging_and_supersession_rules() -> None:
    receipt = _load_json("bridge/templates/prompt_receipt_v1.json")
    receipt["archive_listing"] = ["one.md", "two.md"]
    receipt["archive_member_count"] = 2
    _assert_invalid(
        "bridge/templates/prompt_receipt_v1.schema.json",
        receipt,
        "archive_listing",
    )

    receipt = _load_json("bridge/templates/prompt_receipt_v1.json")
    receipt["supersedes_receipt"] = receipt["receipt_id"]
    with pytest.raises(ValueError, match="supersede itself"):
        _validate_prompt_receipt_semantics(receipt, repo_root=ROOT)

    receipt = _load_json("bridge/templates/prompt_receipt_v1.json")
    receipt["status"] = "active"
    receipt["supersedes_receipt"] = "older-receipt"
    with pytest.raises(ValueError, match="superseded receipt"):
        _validate_prompt_receipt_semantics(receipt, repo_root=ROOT)


def test_active_prompt_receipt_requires_existing_package_and_matching_hash() -> None:
    receipt = _load_json("bridge/templates/prompt_receipt_v1.json")
    receipt["status"] = "active"
    receipt["work_package_path"] = "work_packages/missing.md"
    with pytest.raises(ValueError, match="missing work package"):
        _validate_prompt_receipt_semantics(receipt, repo_root=ROOT)

    receipt = _load_json("bridge/templates/prompt_receipt_v1.json")
    receipt["status"] = "active"
    receipt["work_package_path"] = "work_packages/templates/work_package_v1.md"
    receipt["work_package_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_prompt_receipt_semantics(receipt, repo_root=ROOT)


def test_active_goal_fixture_references_existing_package_and_receipt() -> None:
    goal = _load_json("work_packages/templates/active_goal_metadata_v1.json")
    _validate_active_goal_semantics(goal, repo_root=ROOT)

    missing = copy.deepcopy(goal)
    missing["approved_work_package"] = "work_packages/missing.md"
    with pytest.raises(ValueError, match="missing approved_work_package"):
        _validate_active_goal_semantics(missing, repo_root=ROOT)


def test_latest_status_projection_uses_non_authoritative_live_contract() -> None:
    status = _load_json("bridge/latest_status.json")
    assert status["record_role"] == "non_authoritative_projection"
    assert status["execution_status"] in {
        "not_initialized",
        "idle",
        "active",
        "running",
        "blocked",
        "decision_ready",
        "complete",
        "handoff_ready",
        "closed",
        "invalid",
    }
    assert status["projection_source"]
    assert status["projection_validity"]


def test_relative_paths_reject_escape_and_records_do_not_define_secret_fields() -> None:
    package = _front_matter("work_packages/templates/work_package_v1.md")
    package["authority_documents"] = ["../secret.md"]
    _assert_invalid(
        "work_packages/templates/work_package_metadata_v1.schema.json",
        package,
        "authority_documents",
    )

    for path in [
        "work_packages/templates/work_package_metadata_v1.schema.json",
        "work_packages/templates/active_goal_metadata_v1.schema.json",
        "bridge/templates/prompt_receipt_v1.schema.json",
        "bridge/templates/latest_status_v1.schema.json",
        "bridge/templates/prompt_receipt_v1.json",
        "bridge/templates/latest_status_v1.json",
        "bridge/latest_status.json",
    ]:
        data = _load_json(path)
        _assert_no_path_escape(data)
        _assert_no_secret_fields(data)


def test_markdown_references_and_agents_pointer_resolve() -> None:
    required = [
        "docs/100_governance-authority/100_governance/EIEL-DOC-000111--eiel-autonomous-operating-model-v1.md",
        "docs/100_governance-authority/100_governance/EIEL-DOC-000112--eiel-autonomous-operating-model-v1-audit-and-implementation-design.md",
        "work_packages/README.md",
        "work_packages/templates/work_package_v1.md",
        "bridge/README.md",
        "bridge/latest_status.json",
        "bridge/prompt_receipts/README.md",
        "bridge/templates/latest_status_v1.json",
        "bridge/templates/prompt_receipt_v1.json",
    ]
    for path in required:
        assert (ROOT / path).exists(), path

    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "docs/100_governance-authority/100_governance/EIEL-DOC-000111--eiel-autonomous-operating-model-v1.md" in agents


def test_one_markdown_zip_convention_is_fixture_tested(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text("# Report\n", encoding="utf-8")
    bundle = tmp_path / "report.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.write(report, arcname=report.name)

    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
    assert len(names) == 1
    assert names[0].endswith(".md")

    bad_bundle = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_bundle, "w") as archive:
        archive.writestr("one.md", "# One\n")
        archive.writestr("two.md", "# Two\n")
    with zipfile.ZipFile(bad_bundle) as archive:
        bad_names = archive.namelist()
    assert not (len(bad_names) == 1 and bad_names[0].endswith(".md"))
