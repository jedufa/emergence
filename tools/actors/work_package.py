"""Small repository-native Work Package contract for EIEL engineering work.

A Work Package is the immutable engineering contract handed from Engineering
Authority to an implementation or review session. This module validates,
normalizes, hashes, and binds the document. It does not plan programmes, assign
work, select models, launch Codex, or accept implementation.
"""

from __future__ import annotations

import copy
import fnmatch
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from tools.actors.canonical import canonical_sha256
from tools.actors.delivery_semantics import delivery_commit as delivery_commit_value
from tools.actors.common import (
    ActorError,
    actor_state_root,
    atomic_write_json,
    file_sha256,
    read_json,
    require_explicit_schema_version,
)
from tools.actors.portfolio import WORK_ID_PREFIX, normalize_string_list, validate_prefixed_id

WORK_PACKAGE_SCHEMA_VERSION = "eiel_work_package_v1"
WORK_PACKAGE_BINDING_SCHEMA_VERSION = "eiel_work_package_binding_v1"
WORK_PACKAGE_CONTRACT_RECEIPT_SCHEMA_VERSION = "eiel_work_package_contract_receipt_v1"
CANONICAL_HASH_KIND_V1 = "canonical_work_package_contract_v1"
CANONICAL_HASH_KIND = "canonical_work_package_contract_v2"
LEGACY_HASH_KIND = "raw_file_sha256_v1"
VERIFICATION_EXECUTION_STRICT = "execution_strict"
VERIFICATION_HISTORICAL_AUDIT = "historical_audit"
AUTHORITY_ROLES = (
    "ARCHITECTURE_AUTHORITY",
    "ENGINEERING_AUTHORITY",
    "IMPLEMENTATION_AUTHORITY",
    "REVIEW_AUTHORITY",
)
WORK_PACKAGE_DELIVERY_AUTHORITIES = (
    "ARCHITECTURE_AUTHORITY",
    "ENGINEERING_AUTHORITY",
    "IMPLEMENTATION_AUTHORITY",
)
UNCERTAINTY_CLASSES = ("U1", "U2", "U3", "U4", "U5")
UNCERTAINTY_RANK = {value: index for index, value in enumerate(UNCERTAINTY_CLASSES, start=1)}
WORK_PACKAGE_LIST_FIELDS = (
    "scope",
    "owned_files",
    "forbidden_files",
    "dependencies",
    "acceptance_criteria",
    "tests",
    "evidence",
    "stop_conditions",
)
WORK_PACKAGE_OPTIONAL_CONTRACT_FIELDS = (
    "read_contract",
    "output_policy",
    "execution_budget",
    "decision_basis",
)
READ_CONTRACT_FIELDS = {
    "entry_files",
    "required_documents",
    "search_roots",
    "excluded_roots",
    "known_symbols",
    "maximum_initial_files",
    "broad_search_allowed",
}
OUTPUT_POLICY_FIELDS = {
    "default_output_mode",
    "maximum_display_lines",
    "test_output",
    "git_output",
    "json_output",
    "log_output",
    "recursive_output_requires_reason",
}
DECISION_BASIS_FIELDS = {
    "reason_summary",
    "unresolved_questions",
    "rejected_alternatives",
    "source_decision_refs",
}

EXECUTION_BUDGET_FIELDS = {
    "expected_tool_calls",
    "handoff_recommended_after_tool_calls",
    "maximum_compactions",
    "expected_duration_minutes",
    "context_health_policy",
}
DEFAULT_READ_CONTRACT = {
    "entry_files": [],
    "required_documents": [],
    "search_roots": [],
    "excluded_roots": [],
    "known_symbols": [],
    "maximum_initial_files": 12,
    "broad_search_allowed": False,
}
DEFAULT_OUTPUT_POLICY = {
    "default_output_mode": "compact",
    "maximum_display_lines": 120,
    "test_output": "failures_and_summary",
    "git_output": "short_status_or_stat",
    "json_output": "selected_fields",
    "log_output": "tail_only",
    "recursive_output_requires_reason": True,
}
DEFAULT_EXECUTION_BUDGET = {
    "expected_tool_calls": 25,
    "handoff_recommended_after_tool_calls": 50,
    "maximum_compactions": 1,
    "expected_duration_minutes": 60,
    "context_health_policy": "advisory",
}
# Frozen defaults are part of ``canonical_work_package_contract_v2``.
# Operational creation defaults may evolve, but the historical v2 digest must
# never acquire new meaning because a mutable module constant changed.
V2_FROZEN_READ_CONTRACT_DEFAULTS = {
    "entry_files": [],
    "required_documents": [],
    "search_roots": [],
    "excluded_roots": [],
    "known_symbols": [],
    "maximum_initial_files": 12,
    "broad_search_allowed": False,
}
V2_FROZEN_OUTPUT_POLICY_DEFAULTS = {
    "default_output_mode": "compact",
    "maximum_display_lines": 120,
    "test_output": "failures_and_summary",
    "git_output": "short_status_or_stat",
    "json_output": "selected_fields",
    "log_output": "tail_only",
    "recursive_output_requires_reason": True,
}
V2_FROZEN_EXECUTION_BUDGET_DEFAULTS = {
    "expected_tool_calls": 25,
    "handoff_recommended_after_tool_calls": 50,
    "maximum_compactions": 1,
    "expected_duration_minutes": 60,
    "context_health_policy": "advisory",
}
WORK_PACKAGE_FIELDS = {
    "schema_version",
    "work_id",
    "package_version",
    "title",
    "goal",
    *WORK_PACKAGE_LIST_FIELDS,
    "recommended_authority",
    "recommended_model",
    "uncertainty_class",
    *WORK_PACKAGE_OPTIONAL_CONTRACT_FIELDS,
}
WORK_PACKAGE_REQUIRED_FIELDS = {
    "schema_version",
    "work_id",
    "package_version",
    "goal",
    *WORK_PACKAGE_LIST_FIELDS,
    "recommended_authority",
    "recommended_model",
    "uncertainty_class",
}
# These fields define the engineering contract. A change after execution begins
# requires a new package path and a higher package_version. Title and current
# model recommendation are intentionally advisory/non-material.
WORK_PACKAGE_MATERIAL_FIELDS = (
    "schema_version",
    "work_id",
    "package_version",
    "goal",
    "scope",
    "owned_files",
    "forbidden_files",
    "dependencies",
    "acceptance_criteria",
    "tests",
    "evidence",
    "stop_conditions",
    "recommended_authority",
    "uncertainty_class",
)
WORK_PACKAGE_OPTIONAL_MATERIAL_FIELDS = WORK_PACKAGE_OPTIONAL_CONTRACT_FIELDS
WORK_PACKAGE_BINDING_FIELDS = {
    "schema_version",
    "path",
    "path_kind",
    "bound_repo_root",
    "sha256",
    "sha256_kind",
    "source_sha256",
    "document_sha256",
    "source_commit",
    "package_schema_version",
    "package_version",
    "work_id",
    "recommended_authority",
    "recommended_model",
    "uncertainty_class",
    "contract_receipt_path",
    "contract_receipt_locator",
    "contract_receipt_sha256",
}
WORK_PACKAGE_BINDING_REQUIRED_FIELDS = {
    "schema_version",
    "path",
    "path_kind",
    "bound_repo_root",
    "sha256",
    "sha256_kind",
    "source_sha256",
    "document_sha256",
    "source_commit",
    "package_schema_version",
    "package_version",
    "work_id",
    "recommended_authority",
    "recommended_model",
    "uncertainty_class",
}


def normalize_work_package_chain(raw: Any, *, fallback: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Normalize an ordered Work Item package chain.

    The chain is Work Manager metadata around the existing immutable package
    bindings.  A wrapper is used for the optional exceptional intermediate
    review gate; old records with only ``work_package_binding`` are projected
    as a one-package chain without rewriting their history.
    """

    values = raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)) else []
    if not values and fallback:
        values = [fallback]
    result: list[dict[str, Any]] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, Mapping):
            raise ActorError("work_package_chain entries must be objects", code="work_package_chain_invalid")
        wrapper = dict(value)
        binding_value = wrapper.get("binding") if isinstance(wrapper.get("binding"), Mapping) else wrapper
        binding = normalize_work_package_binding(binding_value)
        if binding is None:
            raise ActorError("work_package_chain entry is missing a binding", code="work_package_chain_invalid")
        requires_review = wrapper.get("requires_intermediate_review", False)
        if not isinstance(requires_review, bool):
            raise ActorError(
                "work_package_chain requires_intermediate_review must be boolean",
                code="work_package_chain_invalid",
            )
        result.append({
            "sequence": int(wrapper.get("sequence") or index),
            "binding": binding,
            "requires_intermediate_review": requires_review,
        })
    sequences = [int(row["sequence"]) for row in result]
    if sequences != list(range(1, len(result) + 1)):
        raise ActorError(
            "work_package_chain sequence must be contiguous and ordered from 1",
            code="work_package_chain_invalid",
        )
    digests = [str(dict(row["binding"]).get("sha256") or "") for row in result]
    if len(digests) != len(set(digests)):
        raise ActorError("work_package_chain contains duplicate package bindings", code="work_package_chain_invalid")
    return result


def work_package_chain_bindings(raw: Any, *, fallback: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    return [dict(row["binding"]) for row in normalize_work_package_chain(raw, fallback=fallback)]
WORK_PACKAGE_SET_LIKE_FIELDS = {
    "owned_files",
    "forbidden_files",
    "dependencies",
    "tests",
    "evidence",
}
MAX_WORK_PACKAGE_BYTES = 256 * 1024
MAX_WORK_PACKAGE_LIST_ITEMS = 256
MAX_WORK_PACKAGE_STRING_LENGTH = 4096
MAX_WORK_PACKAGE_TOTAL_LIST_ITEMS = 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GLOB_RE = re.compile(r"[*?\[]")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[/\\]")


def _duplicate_key_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ActorError(
                f"work package contains duplicate key: {key}",
                code="work_package_duplicate_key",
            )
        value[key] = item
    return value


def _read_json_strict(source: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_key_object,
        )
    except ActorError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActorError(
            f"cannot read work package: {source}: {exc}",
            code="work_package_parse_failed",
        ) from exc
    if not isinstance(value, dict):
        raise ActorError("work package must be a JSON object", code="work_package_invalid")
    return dict(value)


def _read_yaml_strict(source: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ActorError("PyYAML is required to read YAML Work Packages") from exc

    class StrictWorkPackageLoader(yaml.SafeLoader):
        pass

    # Do not let YAML 1.1 turn yes/on/date-like values into booleans or dates.
    # Work Package scalars remain strings except for an unambiguous decimal
    # package_version and explicit null.
    StrictWorkPackageLoader.yaml_implicit_resolvers = copy.deepcopy(
        yaml.SafeLoader.yaml_implicit_resolvers
    )
    rejected_tags = {
        "tag:yaml.org,2002:bool",
        "tag:yaml.org,2002:timestamp",
        "tag:yaml.org,2002:int",
        "tag:yaml.org,2002:float",
        "tag:yaml.org,2002:null",
    }
    for first, resolvers in list(StrictWorkPackageLoader.yaml_implicit_resolvers.items()):
        StrictWorkPackageLoader.yaml_implicit_resolvers[first] = [
            item for item in resolvers if item[0] not in rejected_tags
        ]
    StrictWorkPackageLoader.add_implicit_resolver(
        "tag:yaml.org,2002:int",
        re.compile(r"^(?:0|[1-9][0-9]*)$"),
        list("0123456789"),
    )
    StrictWorkPackageLoader.add_implicit_resolver(
        "tag:yaml.org,2002:null",
        re.compile(r"^(?:null|NULL|Null)$"),
        ["n", "N"],
    )

    def compose_node(loader: Any, parent: Any, index: Any) -> Any:
        if loader.check_event(yaml.AliasEvent):
            event = loader.peek_event()
            raise ActorError(
                f"YAML aliases are forbidden in Work Packages: *{event.anchor}",
                code="work_package_yaml_alias_forbidden",
            )
        return yaml.SafeLoader.compose_node(loader, parent, index)

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[str, Any]:
        if not isinstance(node, yaml.MappingNode):
            raise ActorError("work package YAML root must be a mapping", code="work_package_invalid")
        mapping: dict[str, Any] = {}
        for key_node, value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge" or str(key_node.value) == "<<":
                raise ActorError(
                    "YAML merge keys are forbidden in Work Packages",
                    code="work_package_yaml_merge_forbidden",
                )
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ActorError(
                    "work package YAML keys must be strings",
                    code="work_package_invalid",
                )
            if key in mapping:
                raise ActorError(
                    f"work package contains duplicate key: {key}",
                    code="work_package_duplicate_key",
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    StrictWorkPackageLoader.compose_node = compose_node  # type: ignore[assignment]
    StrictWorkPackageLoader.construct_mapping = construct_mapping  # type: ignore[assignment]

    try:
        value = yaml.load(source.read_text(encoding="utf-8"), Loader=StrictWorkPackageLoader)
    except ActorError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ActorError(
            f"cannot read work package: {source}: {exc}",
            code="work_package_parse_failed",
        ) from exc
    if not isinstance(value, dict):
        raise ActorError("work package must be a YAML object", code="work_package_invalid")
    return dict(value)


def read_work_package_document(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise ActorError(f"cannot inspect work package: {source}: {exc}", code="work_package_parse_failed") from exc
    if size > MAX_WORK_PACKAGE_BYTES:
        raise ActorError(
            f"work package exceeds the {MAX_WORK_PACKAGE_BYTES}-byte size limit: {source}",
            code="work_package_too_large",
        )
    suffix = source.suffix.lower()
    if suffix == ".json":
        return _read_json_strict(source)
    if suffix in {".yaml", ".yml"}:
        return _read_yaml_strict(source)
    raise ActorError(
        f"unsupported Work Package type: {source.suffix or '<none>'}",
        code="work_package_type_unsupported",
    )


def _non_empty_strings(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ActorError(
            f"work package {field} must be a list",
            code="work_package_invalid",
            problems=[f"{field} must contain strings"],
        )
    if any(not isinstance(item, str) for item in value):
        raise ActorError(
            f"work package {field} must contain strings only",
            code="work_package_invalid",
        )
    rows = normalize_string_list(value)
    if len(rows) != len(value):
        raise ActorError(
            f"work package {field} contains empty or duplicate values",
            code="work_package_invalid",
            problems=[f"{field} must contain unique non-empty strings"],
        )
    if len(rows) > MAX_WORK_PACKAGE_LIST_ITEMS:
        raise ActorError(
            f"work package {field} exceeds {MAX_WORK_PACKAGE_LIST_ITEMS} entries",
            code="work_package_too_large",
        )
    oversized = [item for item in rows if len(item) > MAX_WORK_PACKAGE_STRING_LENGTH]
    if oversized:
        raise ActorError(
            f"work package {field} contains text longer than {MAX_WORK_PACKAGE_STRING_LENGTH} characters",
            code="work_package_too_large",
        )
    return rows


def _strict_optional_mapping(value: Any, *, field: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ActorError(f"work package {field} must be an object", code="work_package_invalid")
    mapping = dict(value)
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise ActorError(
            f"work package {field} contains unsupported fields: " + ", ".join(unknown),
            code="work_package_invalid",
        )
    return mapping


def _strict_integer(value: Any, *, field: str, minimum: int, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ActorError(f"work package {field} must be an integer", code="work_package_invalid")
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and at most {maximum}" if maximum is not None else ""
        raise ActorError(
            f"work package {field} must be at least {minimum}{upper}",
            code="work_package_invalid",
        )
    return int(value)


def _strict_boolean(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ActorError(f"work package {field} must be a boolean", code="work_package_invalid")


def normalize_read_contract(raw: Any) -> dict[str, Any]:
    value = _strict_optional_mapping(raw, field="read_contract", allowed=READ_CONTRACT_FIELDS)
    result = dict(DEFAULT_READ_CONTRACT)
    for field in ("entry_files", "required_documents", "search_roots", "excluded_roots", "known_symbols"):
        if field in value:
            result[field] = _non_empty_strings(value.get(field), field=f"read_contract.{field}")
    if "maximum_initial_files" in value:
        result["maximum_initial_files"] = _strict_integer(
            value["maximum_initial_files"],
            field="read_contract.maximum_initial_files",
            minimum=1,
            maximum=100,
        )
    if "broad_search_allowed" in value:
        result["broad_search_allowed"] = _strict_boolean(
            value["broad_search_allowed"], field="read_contract.broad_search_allowed"
        )
    if len(result["entry_files"]) > int(result["maximum_initial_files"]):
        raise ActorError(
            "work package read_contract.entry_files exceeds maximum_initial_files",
            code="work_package_invalid",
        )
    return result


def normalize_output_policy(raw: Any) -> dict[str, Any]:
    value = _strict_optional_mapping(raw, field="output_policy", allowed=OUTPUT_POLICY_FIELDS)
    result = dict(DEFAULT_OUTPUT_POLICY)
    enums = {
        "default_output_mode": {"compact", "normal"},
        "test_output": {"failures_and_summary", "summary", "full"},
        "git_output": {"short_status_or_stat", "name_status", "full"},
        "json_output": {"selected_fields", "summary", "full"},
        "log_output": {"tail_only", "summary", "full"},
    }
    for field, allowed in enums.items():
        if field in value:
            if not isinstance(value[field], str) or value[field].strip().lower() not in allowed:
                raise ActorError(
                    f"work package output_policy.{field} must be one of " + ", ".join(sorted(allowed)),
                    code="work_package_invalid",
                )
            result[field] = value[field].strip().lower()
    if "maximum_display_lines" in value:
        result["maximum_display_lines"] = _strict_integer(
            value["maximum_display_lines"],
            field="output_policy.maximum_display_lines",
            minimum=20,
            maximum=2000,
        )
    if "recursive_output_requires_reason" in value:
        result["recursive_output_requires_reason"] = _strict_boolean(
            value["recursive_output_requires_reason"], field="output_policy.recursive_output_requires_reason"
        )
    return result


def normalize_execution_budget(raw: Any) -> dict[str, Any]:
    value = _strict_optional_mapping(raw, field="execution_budget", allowed=EXECUTION_BUDGET_FIELDS)
    result = dict(DEFAULT_EXECUTION_BUDGET)
    integer_fields = {
        "expected_tool_calls": (0, 10000),
        "handoff_recommended_after_tool_calls": (1, 10000),
        "maximum_compactions": (0, 100),
        "expected_duration_minutes": (1, 10080),
    }
    for field, bounds in integer_fields.items():
        if field in value:
            result[field] = _strict_integer(
                value[field], field=f"execution_budget.{field}", minimum=bounds[0], maximum=bounds[1]
            )
    if result["handoff_recommended_after_tool_calls"] < result["expected_tool_calls"]:
        raise ActorError(
            "work package execution_budget handoff threshold may not be below expected_tool_calls",
            code="work_package_invalid",
        )
    if "context_health_policy" in value:
        policy = str(value["context_health_policy"]).strip().lower()
        if policy != "advisory":
            raise ActorError(
                "work package execution_budget.context_health_policy currently supports advisory only",
                code="work_package_invalid",
            )
        result["context_health_policy"] = policy
    return result


def normalize_decision_basis(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    unknown = sorted(set(value) - DECISION_BASIS_FIELDS)
    if unknown:
        raise ActorError(
            "work package decision_basis contains unsupported fields",
            code="work_package_invalid",
            problems=unknown,
        )
    reason = str(value.get("reason_summary") or "").strip()
    if not reason:
        raise ActorError(
            "work package decision_basis.reason_summary is required",
            code="work_package_invalid",
        )
    alternatives = []
    for row in value.get("rejected_alternatives") or []:
        if not isinstance(row, Mapping):
            raise ActorError("rejected alternative must be an object", code="work_package_invalid")
        item = {
            "alternative": str(row.get("alternative") or "").strip(),
            "reason_rejected": str(row.get("reason_rejected") or "").strip(),
            "evidence_ref": str(row.get("evidence_ref") or "").strip() or None,
        }
        if not item["alternative"] or not item["reason_rejected"]:
            raise ActorError(
                "rejected alternative requires alternative and reason_rejected",
                code="work_package_invalid",
            )
        alternatives.append(item)
    return {
        "reason_summary": reason,
        "unresolved_questions": normalize_string_list(value.get("unresolved_questions")),
        "rejected_alternatives": alternatives,
        "source_decision_refs": normalize_string_list(value.get("source_decision_refs")),
    }


def effective_read_contract(package: Mapping[str, Any]) -> dict[str, Any]:
    value = package.get("read_contract")
    return normalize_read_contract(value) if value is not None else dict(DEFAULT_READ_CONTRACT)


def effective_output_policy(package: Mapping[str, Any]) -> dict[str, Any]:
    value = package.get("output_policy")
    return normalize_output_policy(value) if value is not None else dict(DEFAULT_OUTPUT_POLICY)


def effective_execution_budget(package: Mapping[str, Any]) -> dict[str, Any]:
    value = package.get("execution_budget")
    return normalize_execution_budget(value) if value is not None else dict(DEFAULT_EXECUTION_BUDGET)


def normalize_work_package(raw: Mapping[str, Any]) -> dict[str, Any]:
    package = dict(raw)
    for field in ("schema_version", "work_id", "goal", "recommended_authority", "uncertainty_class"):
        if field in package and not isinstance(package[field], str):
            raise ActorError(f"work package {field} must be a string", code="work_package_invalid")
    for field in ("title", "recommended_model"):
        if field in package and package[field] is not None and not isinstance(package[field], str):
            raise ActorError(f"work package {field} must be a string or null", code="work_package_invalid")
    if "package_version" in package and (
        not isinstance(package["package_version"], int) or isinstance(package["package_version"], bool)
    ):
        raise ActorError("work package package_version must be an integer", code="work_package_invalid")
    package["schema_version"] = str(package.get("schema_version") or WORK_PACKAGE_SCHEMA_VERSION).strip()
    package["work_id"] = validate_prefixed_id(
        str(package.get("work_id") or ""), WORK_ID_PREFIX, label="work package work item"
    )
    package["package_version"] = int(package.get("package_version", 1))
    package["title"] = str(package.get("title") or "").strip() or None
    package["goal"] = str(package.get("goal") or "").strip()
    if len(package["goal"]) > MAX_WORK_PACKAGE_STRING_LENGTH:
        raise ActorError("work package goal is too long", code="work_package_too_large")
    if package["title"] and len(str(package["title"])) > 512:
        raise ActorError("work package title is too long", code="work_package_too_large")
    for field in WORK_PACKAGE_LIST_FIELDS:
        package[field] = _non_empty_strings(package.get(field), field=field)
    package["dependencies"] = [
        validate_prefixed_id(value, WORK_ID_PREFIX, label="work package dependency")
        for value in package["dependencies"]
    ]
    package["recommended_authority"] = str(
        package.get("recommended_authority") or "IMPLEMENTATION_AUTHORITY"
    ).strip().upper()
    package["recommended_model"] = str(package.get("recommended_model") or "").strip() or None
    package["uncertainty_class"] = str(package.get("uncertainty_class") or "U2").strip().upper()
    if sum(len(package[field]) for field in WORK_PACKAGE_LIST_FIELDS) > MAX_WORK_PACKAGE_TOTAL_LIST_ITEMS:
        raise ActorError(
            f"work package exceeds {MAX_WORK_PACKAGE_TOTAL_LIST_ITEMS} total contract-list entries",
            code="work_package_too_large",
        )
    if "read_contract" in package and package.get("read_contract") is not None:
        package["read_contract"] = normalize_read_contract(package["read_contract"])
    else:
        package.pop("read_contract", None)
    if "output_policy" in package and package.get("output_policy") is not None:
        package["output_policy"] = normalize_output_policy(package["output_policy"])
    else:
        package.pop("output_policy", None)
    if "execution_budget" in package and package.get("execution_budget") is not None:
        package["execution_budget"] = normalize_execution_budget(package["execution_budget"])
    else:
        package.pop("execution_budget", None)
    if "decision_basis" in package and package.get("decision_basis") is not None:
        if not isinstance(package["decision_basis"], Mapping):
            raise ActorError("work package decision_basis must be an object", code="work_package_invalid")
        package["decision_basis"] = normalize_decision_basis(package["decision_basis"])
    else:
        package.pop("decision_basis", None)
    return package


def validate_work_package(
    package: Mapping[str, Any],
    *,
    expected_work_id: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(field for field in WORK_PACKAGE_REQUIRED_FIELDS if field not in package)
    unknown = sorted(str(field) for field in package if field not in WORK_PACKAGE_FIELDS)
    if missing:
        errors.append("work package is missing required fields: " + ", ".join(missing))
    if unknown:
        errors.append("work package contains unsupported fields: " + ", ".join(unknown))
    try:
        normalized = normalize_work_package(package)
    except ActorError as exc:
        return {
            "valid": False,
            "errors": [str(exc), *list(exc.problems or [])],
            "warnings": list(exc.warnings or []),
            "package": None,
        }
    if normalized["schema_version"] != WORK_PACKAGE_SCHEMA_VERSION:
        errors.append("unsupported work package schema_version")
    if normalized["package_version"] < 1:
        errors.append("work package package_version must be at least 1")
    if not normalized["goal"]:
        errors.append("work package goal is required")
    if normalized["recommended_authority"] not in WORK_PACKAGE_DELIVERY_AUTHORITIES:
        errors.append(
            "recommended_authority must be one of " + ", ".join(WORK_PACKAGE_DELIVERY_AUTHORITIES)
            + "; Review Authority sessions target a completed delivery under the same package"
        )
    if normalized["uncertainty_class"] not in UNCERTAINTY_CLASSES:
        errors.append("uncertainty_class must be one of " + ", ".join(UNCERTAINTY_CLASSES))
    if normalized["work_id"] in normalized["dependencies"]:
        errors.append("work package cannot depend on its own work_id")
    if expected_work_id and normalized["work_id"] != str(expected_work_id).strip().upper():
        errors.append(
            f"work package work_id {normalized['work_id']} does not match {str(expected_work_id).strip().upper()}"
        )
    for required_field in ("scope", "owned_files", "acceptance_criteria", "tests", "evidence", "stop_conditions"):
        if not normalized[required_field]:
            errors.append(f"work package {required_field} must contain at least one bounded entry")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "package": normalized,
    }


def load_work_package_value(raw: Any, *, expected_work_id: str | None = None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError("Work Package must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        raw, expected=WORK_PACKAGE_SCHEMA_VERSION, label="Work Package"
    )
    validation = validate_work_package(raw, expected_work_id=expected_work_id)
    if not validation["valid"]:
        raise ActorError(
            "work package validation failed",
            code="work_package_invalid",
            problems=validation["errors"],
            warnings=validation["warnings"],
        )
    return dict(validation["package"])


def load_work_package(path: str | Path, *, expected_work_id: str | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ActorError(f"work package does not exist: {source}", code="work_package_missing")
    raw = read_work_package_document(source)
    return load_work_package_value(raw, expected_work_id=expected_work_id)


def work_package_material_payload(
    package: Mapping[str, Any],
    *,
    include_version: bool = True,
    hash_kind: str = CANONICAL_HASH_KIND,
) -> dict[str, Any]:
    normalized = normalize_work_package(package)
    fields = WORK_PACKAGE_MATERIAL_FIELDS if include_version else tuple(
        field for field in WORK_PACKAGE_MATERIAL_FIELDS if field != "package_version"
    )
    payload = {field: normalized.get(field) for field in fields}
    if hash_kind == CANONICAL_HASH_KIND_V1:
        for field in WORK_PACKAGE_OPTIONAL_MATERIAL_FIELDS:
            if field in normalized:
                payload[field] = normalized[field]
        return payload
    if hash_kind != CANONICAL_HASH_KIND:
        raise ActorError(f"unsupported Work Package contract hash kind: {hash_kind}", code="work_package_hash_kind_invalid")
    for field in WORK_PACKAGE_SET_LIKE_FIELDS:
        if field in payload:
            payload[field] = sorted(payload[field])
    payload["read_contract"] = (
        normalize_read_contract(normalized["read_contract"])
        if "read_contract" in normalized
        else copy.deepcopy(V2_FROZEN_READ_CONTRACT_DEFAULTS)
    )
    payload["output_policy"] = (
        normalize_output_policy(normalized["output_policy"])
        if "output_policy" in normalized
        else copy.deepcopy(V2_FROZEN_OUTPUT_POLICY_DEFAULTS)
    )
    payload["execution_budget"] = (
        normalize_execution_budget(normalized["execution_budget"])
        if "execution_budget" in normalized
        else copy.deepcopy(V2_FROZEN_EXECUTION_BUDGET_DEFAULTS)
    )
    if "decision_basis" in normalized:
        payload["decision_basis"] = normalized["decision_basis"]
    return payload


def work_package_contract_sha256(
    package: Mapping[str, Any], *, hash_kind: str = CANONICAL_HASH_KIND
) -> str:
    return canonical_sha256(work_package_material_payload(package, hash_kind=hash_kind))


def work_package_document_sha256(package: Mapping[str, Any]) -> str:
    return canonical_sha256(normalize_work_package(package))


def work_package_material_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return work_package_material_payload(left, include_version=False) == work_package_material_payload(
        right, include_version=False
    )


def _repository_relative_path(path: str | Path, *, repo_root: str | Path) -> tuple[Path, str]:
    repo = Path(repo_root).expanduser().resolve()
    source = Path(path).expanduser()
    candidate = source if source.is_absolute() else repo / source
    if candidate.is_symlink():
        raise ActorError(
            f"work package may not be a symbolic link: {candidate}",
            code="work_package_symlink_forbidden",
        )
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(repo).as_posix()
    except ValueError as exc:
        raise ActorError(
            f"work package must be inside the repository: {resolved}",
            code="work_package_outside_repository",
            problems=[f"repository root: {repo}"],
        ) from exc
    _assert_no_symlink_components(repo, PurePosixPath(relative).parts, label="Work Package")
    return resolved, relative


def _assert_exact_case(parent: Path, component: str, *, label: str) -> None:
    if not parent.is_dir():
        return
    try:
        names = [entry.name for entry in parent.iterdir()]
    except OSError as exc:
        raise ActorError(f"cannot inspect {label} path parent {parent}: {exc}", code="work_package_path_invalid") from exc
    if component in names:
        return
    folded = [name for name in names if name.casefold() == component.casefold()]
    if folded:
        raise ActorError(
            f"{label} path case does not match repository entry: {component} (observed {folded[0]})",
            code="work_package_path_case_mismatch",
        )


def _assert_no_symlink_components(repo: Path, parts: Sequence[str], *, label: str) -> None:
    current = repo
    for part in parts:
        _assert_exact_case(current, part, label=label)
        current = current / part
        if current.is_symlink():
            raise ActorError(
                f"{label} path traverses a symbolic link: {current}",
                code="work_package_path_symlink_forbidden",
            )
        if not current.exists():
            break


def normalize_package_path_entry(value: str, *, repo_root: str | Path, field: str) -> str:
    raw = str(value).strip()
    if not raw:
        raise ActorError(f"{field} contains an empty path", code="work_package_path_invalid")
    if "\x00" in raw or "\\" in raw:
        raise ActorError(
            f"{field} path must use repository-relative POSIX syntax: {raw}",
            code="work_package_path_invalid",
        )
    if raw.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(raw):
        raise ActorError(f"{field} path must be repository-relative: {raw}", code="work_package_path_escape")
    # Validate the operator's literal spelling before PurePosixPath is allowed
    # to normalize it.  Otherwise spellings such as ``a//b`` and ``a/./b``
    # become indistinguishable from canonical input and weaken the path-fence
    # contract the Work Package is meant to provide.
    raw_parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ActorError(
            f"{field} path escapes or ambiguously addresses the repository: {raw}",
            code="work_package_path_escape",
        )
    parts = PurePosixPath(raw).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ActorError(f"{field} path escapes or ambiguously addresses the repository: {raw}", code="work_package_path_escape")
    normalized = PurePosixPath(*parts).as_posix()
    repo = Path(repo_root).expanduser().resolve()
    static_parts: list[str] = []
    for part in parts:
        if _GLOB_RE.search(part):
            break
        static_parts.append(part)
    _assert_no_symlink_components(repo, static_parts, label=field)
    candidate = repo.joinpath(*static_parts)
    try:
        candidate.resolve(strict=False).relative_to(repo)
    except ValueError as exc:
        raise ActorError(f"{field} path escapes the repository: {raw}", code="work_package_path_escape") from exc
    if not _GLOB_RE.search(normalized):
        _assert_no_symlink_components(repo, parts, label=field)
    return normalized


def normalize_package_file_fences(package: Mapping[str, Any], *, repo_root: str | Path) -> dict[str, Any]:
    normalized = normalize_work_package(package)
    for field in ("owned_files", "forbidden_files"):
        normalized[field] = [
            normalize_package_path_entry(value, repo_root=repo_root, field=field)
            for value in normalized[field]
        ]
        if len(set(normalized[field])) != len(normalized[field]):
            raise ActorError(f"{field} contains duplicate normalized paths", code="work_package_path_invalid")
    exact_overlap = sorted(set(normalized["owned_files"]).intersection(normalized["forbidden_files"]))
    if exact_overlap:
        raise ActorError(
            "owned_files and forbidden_files overlap",
            code="work_package_path_fence_conflict",
            problems=exact_overlap,
        )
    if "read_contract" in normalized:
        contract = dict(normalized["read_contract"])
        for field in ("entry_files", "required_documents", "search_roots", "excluded_roots"):
            contract[field] = [
                normalize_package_path_entry(value, repo_root=repo_root, field=f"read_contract.{field}")
                for value in contract[field]
            ]
            if len(set(contract[field])) != len(contract[field]):
                raise ActorError(
                    f"read_contract.{field} contains duplicate normalized paths",
                    code="work_package_path_invalid",
                )
        excluded = contract["excluded_roots"]
        required = [*contract["entry_files"], *contract["required_documents"]]
        conflicts = sorted(
            value
            for value in required
            if any(package_path_matches(value, pattern, repo_root=repo_root) for pattern in excluded)
        )
        if conflicts:
            raise ActorError(
                "read_contract excludes required initial files",
                code="work_package_read_contract_conflict",
                problems=conflicts,
            )
        normalized["read_contract"] = contract
    return normalized


def _segment_glob_match(path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]) -> bool:
    cache: dict[tuple[int, int], bool] = {}

    def match(path_index: int, pattern_index: int) -> bool:
        key = (path_index, pattern_index)
        if key in cache:
            return cache[key]
        if pattern_index == len(pattern_parts):
            result = path_index == len(path_parts)
        elif pattern_parts[pattern_index] == "**":
            result = match(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and match(path_index + 1, pattern_index)
            )
        elif path_index >= len(path_parts):
            result = False
        else:
            result = fnmatch.fnmatchcase(path_parts[path_index], pattern_parts[pattern_index]) and match(
                path_index + 1, pattern_index + 1
            )
        cache[key] = result
        return result

    return match(0, 0)


def package_path_matches(relative_path: str, pattern: str, *, repo_root: str | Path | None = None) -> bool:
    relative = PurePosixPath(str(relative_path)).as_posix()
    normalized_pattern = PurePosixPath(str(pattern)).as_posix()
    if _GLOB_RE.search(normalized_pattern):
        return _segment_glob_match(
            PurePosixPath(relative).parts, PurePosixPath(normalized_pattern).parts
        )
    return relative == normalized_pattern


def validate_changed_paths(
    changed_files: Sequence[str],
    package: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    normalized = normalize_package_file_fences(package, repo_root=repo_root)
    owned = normalized["owned_files"]
    forbidden = normalized["forbidden_files"]
    outside: list[str] = []
    forbidden_hits: list[str] = []
    normalized_changed: list[str] = []
    for value in changed_files:
        path = normalize_package_path_entry(value, repo_root=repo_root, field="changed_files")
        normalized_changed.append(path)
        if any(package_path_matches(path, pattern, repo_root=repo_root) for pattern in forbidden):
            forbidden_hits.append(path)
        if not any(package_path_matches(path, pattern, repo_root=repo_root) for pattern in owned):
            outside.append(path)
    return {
        "valid": not outside and not forbidden_hits,
        "changed_files": sorted(set(normalized_changed)),
        "outside_owned_files": sorted(set(outside)),
        "forbidden_files_changed": sorted(set(forbidden_hits)),
    }


def _git(repo_root: str | Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(Path(repo_root).expanduser().resolve()), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def delivery_recorded_at(record: Mapping[str, Any]) -> str:
    payload = dict(record.get("payload") or {})
    evidence = dict(payload.get("delivery_evidence") or {})
    return str(
        evidence.get("recorded_at_utc")
        or record.get("created_at_utc")
        or ""
    )


def review_recorded_at(record: Mapping[str, Any]) -> str:
    payload = dict(record.get("payload") or {})
    evidence = dict(payload.get("review_evidence") or {})
    return str(
        evidence.get("reviewed_at_utc")
        or record.get("created_at_utc")
        or ""
    )


def _delivery_commit(record: Mapping[str, Any]) -> str:
    payload = dict(record.get("payload") or {})
    return delivery_commit_value(dict(payload.get("delivery_evidence") or {}))


def _delivery_repo_roots(record: Mapping[str, Any]) -> list[str]:
    payload = dict(record.get("payload") or {})
    roots = [
        str(dict(payload.get("repository") or {}).get("repo_root") or "").strip(),
        str(dict(payload.get("work_package_binding") or {}).get("bound_repo_root") or "").strip(),
    ]
    return [value for index, value in enumerate(roots) if value and value not in roots[:index]]


def _retained_delivery_proof(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
) -> dict[str, Any] | None:
    payload = dict(record.get("payload") or {})
    evidence = dict(payload.get("delivery_evidence") or {})
    reference = evidence.get("delivery_proof")
    if not isinstance(reference, Mapping):
        return None
    try:
        from tools.actors.repository_truth import (
            DELIVERY_PROOF_SCHEMA_VERSION,
            load_repository_truth_artifact,
        )

        return load_repository_truth_artifact(
            reference,
            state_root=state_root,
            expected_schema_version=DELIVERY_PROOF_SCHEMA_VERSION,
        )
    except ActorError:
        return None


def _commit_relation(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    state_root: str | Path | None = None,
) -> str:
    left_commit = _delivery_commit(left)
    right_commit = _delivery_commit(right)
    if not left_commit or not right_commit:
        return "unknown"
    if left_commit == right_commit:
        return "same"

    # Current delivery evidence carries a durable commit graph and binary patch.
    # Prefer it to live Git so old delivery authority remains reconstructable
    # after branch deletion, garbage collection, archive transfer, or checkout
    # relocation.  Live Git is a compatibility fallback for legacy records only.
    left_proof = _retained_delivery_proof(left, state_root=state_root)
    right_proof = _retained_delivery_proof(right, state_root=state_root)
    if left_proof is not None and right_proof is not None:
        from tools.actors.repository_truth import persisted_commit_relation

        return persisted_commit_relation(left_proof, right_proof)

    roots = [*_delivery_repo_roots(left), *_delivery_repo_roots(right)]
    roots = [value for index, value in enumerate(roots) if value not in roots[:index]]
    for root in roots:
        if not Path(root).exists():
            continue
        left_to_right = _git(root, "merge-base", "--is-ancestor", left_commit, right_commit)
        if left_to_right.returncode == 0:
            return "left_ancestor"
        right_to_left = _git(root, "merge-base", "--is-ancestor", right_commit, left_commit)
        if right_to_left.returncode == 0:
            return "right_ancestor"
        if left_to_right.returncode == 1 and right_to_left.returncode == 1:
            return "divergent"
    return "unknown"


def resolve_delivery_lineage(
    records: Sequence[Mapping[str, Any]],
    *,
    state_root: str | Path | None = None,
) -> dict[str, Any]:
    ordered = sorted(
        (dict(row) for row in records),
        key=lambda row: (delivery_recorded_at(row), str(row.get("record_id") or "")),
        reverse=True,
    )
    if not ordered:
        return {
            "ordered": [],
            "authoritative": None,
            "conflict": False,
            "conflicting_session_ids": [],
            "superseded_session_ids": [],
            "problems": [],
        }
    representatives: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    seen_commits: set[str] = set()
    for row in ordered:
        commit = _delivery_commit(row)
        if commit and commit in seen_commits:
            duplicate_ids.append(str(row.get("record_id") or ""))
            continue
        if commit:
            seen_commits.add(commit)
        representatives.append(row)
    superseded: set[str] = set(duplicate_ids)
    problems: list[str] = []
    for index, left in enumerate(representatives):
        for right in representatives[index + 1 :]:
            relation = _commit_relation(left, right, state_root=state_root)
            left_id = str(left.get("record_id") or "")
            right_id = str(right.get("record_id") or "")
            if relation == "left_ancestor":
                superseded.add(left_id)
            elif relation == "right_ancestor":
                superseded.add(right_id)
            elif relation in {"divergent", "unknown"}:
                problems.append(
                    f"delivery commits are {relation}: {left_id} ({_delivery_commit(left)}) and "
                    f"{right_id} ({_delivery_commit(right)})"
                )
    maximal = [row for row in representatives if str(row.get("record_id") or "") not in superseded]
    conflict = len(maximal) > 1 or bool(problems)
    authoritative = maximal[0] if len(maximal) == 1 and not conflict else None
    return {
        "ordered": ordered,
        "authoritative": authoritative,
        "conflict": conflict,
        "conflicting_session_ids": [str(row.get("record_id") or "") for row in maximal] if conflict else [],
        "superseded_session_ids": sorted(value for value in superseded if value),
        "problems": problems,
    }


def _git_package_provenance(
    source: Path,
    *,
    repo_root: str | Path,
    required: bool,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    relative = source.resolve().relative_to(repo).as_posix()
    top = _git(repo, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        if required:
            raise ActorError(
                "active Work Package binding requires a Git repository",
                code="work_package_git_provenance_missing",
                problems=[top.stderr.strip() or "git rev-parse failed"],
            )
        return {"tracked": False, "source_commit": None}
    tracked = _git(repo, "ls-files", "--error-unmatch", "--", relative)
    status = _git(repo, "status", "--porcelain=v1", "--", relative)
    head = _git(repo, "rev-parse", "HEAD")
    problems: list[str] = []
    if tracked.returncode != 0:
        problems.append(f"Work Package is not tracked by Git: {relative}")
    if status.returncode != 0:
        problems.append(status.stderr.strip() or "git status failed")
    elif status.stdout.strip():
        problems.append(f"Work Package has uncommitted index or worktree changes: {relative}")
    if head.returncode != 0 or not head.stdout.strip():
        problems.append(head.stderr.strip() or "cannot resolve Git HEAD")
    if required and problems:
        raise ActorError(
            "active Work Package must be committed and repository-persistent",
            code="work_package_git_provenance_invalid",
            problems=problems,
        )
    return {
        "tracked": tracked.returncode == 0,
        "source_commit": head.stdout.strip() if head.returncode == 0 else None,
    }


def bind_work_package(
    path: str | Path,
    *,
    repo_root: str | Path,
    expected_work_id: str | None = None,
    require_git_provenance: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source, relative = _repository_relative_path(path, repo_root=repo_root)
    package = load_work_package(source, expected_work_id=expected_work_id)
    package = normalize_package_file_fences(package, repo_root=repo_root)
    provenance = _git_package_provenance(
        source,
        repo_root=repo_root,
        required=require_git_provenance,
    )
    binding = {
        "schema_version": WORK_PACKAGE_BINDING_SCHEMA_VERSION,
        "path": relative,
        "path_kind": "repository_relative",
        "bound_repo_root": str(Path(repo_root).expanduser().resolve()),
        "sha256": work_package_contract_sha256(package),
        "sha256_kind": CANONICAL_HASH_KIND,
        "source_sha256": file_sha256(source),
        "document_sha256": work_package_document_sha256(package),
        "source_commit": provenance.get("source_commit"),
        "package_schema_version": package["schema_version"],
        "package_version": package["package_version"],
        "work_id": package["work_id"],
        "recommended_authority": package["recommended_authority"],
        "recommended_model": package["recommended_model"],
        "uncertainty_class": package["uncertainty_class"],
    }
    return binding, package


def archive_work_package_contract(
    binding: Mapping[str, Any],
    package: Mapping[str, Any],
    *,
    state_root: str | Path | None,
) -> dict[str, Any]:
    """Archive the exact normalized contract used by a binding.

    The repository remains the authoring authority.  This immutable receipt is
    only the durable historical evidence needed when a later checkout no longer
    contains the old package path.
    """

    normalized_package = normalize_work_package(package)
    contract_sha = str(binding.get("sha256") or work_package_contract_sha256(normalized_package))
    receipt = {
        "schema_version": WORK_PACKAGE_CONTRACT_RECEIPT_SCHEMA_VERSION,
        "contract_sha256": contract_sha,
        "document_sha256": work_package_document_sha256(normalized_package),
        "package": normalized_package,
    }
    receipt_sha = canonical_sha256(receipt)
    # A material contract digest intentionally ignores advisory/display-only
    # fields. Key the archived receipt by the complete normalized document so
    # two non-material revisions can coexist without rewriting history.
    document_sha = str(receipt["document_sha256"])
    target = actor_state_root(state_root) / "work-package-contracts" / f"{document_sha}.json"
    if target.exists():
        existing = read_json(target)
        if canonical_sha256(existing) != receipt_sha:
            raise ActorError(
                "historical Work Package contract receipt conflicts with existing evidence",
                code="work_package_contract_receipt_conflict",
                problems=[str(target)],
            )
    else:
        atomic_write_json(target, receipt)
    enriched = dict(binding)
    enriched["contract_receipt_path"] = str(target)
    enriched["contract_receipt_locator"] = f"work-package-contracts/{document_sha}.json"
    enriched["contract_receipt_sha256"] = receipt_sha
    return enriched


def load_work_package_contract_receipt(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError("Work Package contract receipt must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        raw,
        expected=WORK_PACKAGE_CONTRACT_RECEIPT_SCHEMA_VERSION,
        label="Work Package contract receipt",
    )
    package = raw.get("package")
    if not isinstance(package, Mapping):
        raise ActorError(
            "Work Package contract receipt contains no package",
            code="schema_record_malformed",
        )
    normalized = load_work_package_value(package)
    observed_contract = raw.get("contract_sha256")
    actual_contract = work_package_contract_sha256(normalized)
    observed_document = raw.get("document_sha256")
    actual_document = work_package_document_sha256(normalized)
    problems: list[str] = []
    if not isinstance(observed_contract, str) or observed_contract != actual_contract:
        problems.append(
            f"contract_sha256 mismatch: expected {actual_contract}, observed {observed_contract}"
        )
    if not isinstance(observed_document, str) or observed_document != actual_document:
        problems.append(
            f"document_sha256 mismatch: expected {actual_document}, observed {observed_document}"
        )
    if problems:
        raise ActorError(
            "Work Package contract receipt digest mismatch",
            code="work_package_contract_receipt_digest_mismatch",
            problems=problems,
        )
    return dict(raw)



def _load_contract_receipt(
    binding: Mapping[str, Any],
    *,
    state_root: str | Path | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Load immutable package evidence through relocatable content locators."""

    candidates: list[Path] = []
    path_text = str(binding.get("contract_receipt_path") or "").strip()
    if path_text:
        candidates.append(Path(path_text).expanduser())

    state_base = actor_state_root(state_root) if state_root is not None else None
    locator = str(binding.get("contract_receipt_locator") or "").strip()
    if locator and state_base is not None:
        locator_path = PurePosixPath(locator)
        if locator_path.is_absolute() or any(part in {"", ".", ".."} for part in locator_path.parts):
            return None, ["historical Work Package contract receipt locator is invalid"]
        candidate = state_base.joinpath(*locator_path.parts).resolve(strict=False)
        try:
            candidate.relative_to(state_base.resolve())
        except ValueError:
            return None, ["historical Work Package contract receipt locator escapes state root"]
        candidates.append(candidate)

    document_sha = str(binding.get("document_sha256") or "").strip().lower()
    if state_base is not None and _SHA256_RE.fullmatch(document_sha):
        candidates.append(state_base / "work-package-contracts" / f"{document_sha}.json")

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.expanduser().resolve(strict=False))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    existing = [candidate for candidate in unique_candidates if candidate.is_file()]
    if not existing:
        if not unique_candidates:
            return None, ["historical Work Package contract receipt is unavailable"]
        return None, [
            "historical Work Package contract receipt is missing at every retained locator: "
            + ", ".join(str(path) for path in unique_candidates)
        ]

    loaded_receipts: list[tuple[Path, dict[str, Any]]] = []
    for path in existing:
        try:
            receipt = load_work_package_contract_receipt(read_json(path))
        except ActorError as exc:
            return None, [f"{path}: {exc}", *list(exc.problems or [])]
        loaded_receipts.append((path, receipt))

    observed_digests = {canonical_sha256(receipt) for _, receipt in loaded_receipts}
    if len(observed_digests) != 1:
        return None, [
            "historical Work Package contract receipt locators resolve conflicting content: "
            + ", ".join(str(path) for path, _ in loaded_receipts)
        ]
    receipt = loaded_receipts[0][1]
    observed = canonical_sha256(receipt)
    expected = str(binding.get("contract_receipt_sha256") or "")
    if expected and observed != expected:
        return None, [
            f"historical Work Package contract receipt digest changed: expected {expected}, observed {observed}"
        ]
    if document_sha and str(receipt.get("document_sha256") or "") != document_sha:
        return None, ["historical Work Package contract receipt document digest does not match the binding"]
    package = receipt.get("package") if isinstance(receipt, Mapping) else None
    if not isinstance(package, Mapping):
        return None, ["historical Work Package contract receipt contains no package"]
    try:
        require_explicit_schema_version(
            package,
            expected=WORK_PACKAGE_SCHEMA_VERSION,
            label="Work Package",
        )
    except ActorError as exc:
        return None, [str(exc)]
    normalized = normalize_work_package(package)
    if work_package_contract_sha256(
        normalized,
        hash_kind=str(binding.get("sha256_kind") or CANONICAL_HASH_KIND),
    ) != str(binding.get("sha256") or ""):
        return None, ["historical Work Package contract receipt does not match the binding digest"]
    return normalized, []


def normalize_work_package_binding(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ActorError("work_package_binding must be an object", code="schema_record_malformed")
    binding = dict(raw)
    require_explicit_schema_version(
        binding,
        expected=WORK_PACKAGE_BINDING_SCHEMA_VERSION,
        label="Work Package binding",
    )
    binding["schema_version"] = str(binding.get("schema_version")).strip()
    binding["path"] = str(binding.get("path") or "").strip()
    binding["path_kind"] = str(binding.get("path_kind") or "repository_relative").strip()
    binding["bound_repo_root"] = str(binding.get("bound_repo_root") or "").strip()
    binding["sha256"] = str(binding.get("sha256") or "").strip().lower()
    binding["sha256_kind"] = str(binding.get("sha256_kind") or LEGACY_HASH_KIND).strip()
    binding["source_sha256"] = str(
        binding.get("source_sha256") or (binding["sha256"] if binding["sha256_kind"] == LEGACY_HASH_KIND else "")
    ).strip().lower() or None
    binding["document_sha256"] = str(binding.get("document_sha256") or "").strip().lower() or None
    binding["source_commit"] = str(binding.get("source_commit") or "").strip().lower() or None
    binding["contract_receipt_path"] = str(binding.get("contract_receipt_path") or "").strip() or None
    binding["contract_receipt_locator"] = str(binding.get("contract_receipt_locator") or "").strip() or None
    binding["contract_receipt_sha256"] = str(binding.get("contract_receipt_sha256") or "").strip().lower() or None
    binding["package_schema_version"] = str(binding.get("package_schema_version") or "").strip()
    try:
        binding["package_version"] = int(binding.get("package_version", 1))
    except (TypeError, ValueError) as exc:
        raise ActorError("work package binding package_version is invalid", code="work_package_binding_invalid") from exc
    binding["work_id"] = validate_prefixed_id(
        str(binding.get("work_id") or ""), WORK_ID_PREFIX, label="work package binding work item"
    )
    binding["recommended_authority"] = str(
        binding.get("recommended_authority") or "IMPLEMENTATION_AUTHORITY"
    ).strip().upper()
    binding["recommended_model"] = str(binding.get("recommended_model") or "").strip() or None
    binding["uncertainty_class"] = str(binding.get("uncertainty_class") or "U2").strip().upper()
    return binding


def load_work_package_binding(raw: Any) -> dict[str, Any]:
    """Strict read-only Work Package binding reader."""

    normalized = normalize_work_package_binding(raw)
    if normalized is None:
        raise ActorError("Work Package binding is missing", code="schema_record_malformed")
    return normalized


def resolve_work_package_binding(
    binding: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> Path:
    normalized = normalize_work_package_binding(binding)
    if normalized is None:
        raise ActorError("work package binding is missing", code="work_package_binding_missing")
    if normalized["path_kind"] != "repository_relative":
        raise ActorError("unsupported work package path_kind", code="work_package_binding_invalid")
    raw_root = repo_root if repo_root is not None else normalized["bound_repo_root"]
    if not str(raw_root or "").strip():
        raise ActorError("work package binding has no repository root", code="work_package_binding_invalid")
    root = Path(raw_root).expanduser().resolve()
    raw_path = normalized["path"]
    if Path(raw_path).is_absolute() or _WINDOWS_ABSOLUTE_RE.match(raw_path):
        raise ActorError("work package path must be repository-relative", code="work_package_binding_invalid")
    parts = PurePosixPath(raw_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ActorError("work package path escapes the repository", code="work_package_binding_invalid")
    _assert_no_symlink_components(root, parts, label="bound Work Package")
    candidate = root.joinpath(*parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ActorError("work package path escapes the repository", code="work_package_binding_invalid") from exc
    return resolved


def verify_work_package_binding(
    binding: Mapping[str, Any] | None,
    *,
    repo_root: str | Path | None = None,
    expected_work_id: str | None = None,
    verification_mode: str = VERIFICATION_HISTORICAL_AUDIT,
    state_root: str | Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(binding, Mapping):
        return {
            "valid": False,
            "errors": ["work package binding is missing"],
            "warnings": [],
            "package": None,
            "path": None,
            "binding": None,
        }
    try:
        normalized = normalize_work_package_binding(binding)
    except ActorError as exc:
        return {
            "valid": False,
            "errors": [str(exc), *list(exc.problems or [])],
            "warnings": list(exc.warnings or []),
            "package": None,
            "path": None,
            "binding": None,
        }
    assert normalized is not None
    if verification_mode not in {VERIFICATION_EXECUTION_STRICT, VERIFICATION_HISTORICAL_AUDIT}:
        return {
            "valid": False,
            "errors": [f"unsupported Work Package verification mode: {verification_mode}"],
            "warnings": [],
            "package": None,
            "path": None,
            "binding": normalized,
        }
    strict = verification_mode == VERIFICATION_EXECUTION_STRICT
    unknown = sorted(str(field) for field in binding if field not in WORK_PACKAGE_BINDING_FIELDS)
    if unknown:
        errors.append("work package binding contains unsupported fields: " + ", ".join(unknown))
    if normalized["schema_version"] != WORK_PACKAGE_BINDING_SCHEMA_VERSION:
        errors.append("unsupported work package binding schema_version")
    if normalized["package_schema_version"] != WORK_PACKAGE_SCHEMA_VERSION:
        errors.append("unsupported bound work package schema_version")
    if normalized["path_kind"] != "repository_relative":
        errors.append("work package binding path_kind must be repository_relative")
    if not normalized["path"]:
        errors.append("work package binding path is required")
    elif Path(normalized["path"]).is_absolute():
        errors.append("work package binding path must be repository-relative")
    if normalized["package_version"] < 1:
        errors.append("work package binding package_version must be at least 1")
    if not normalized["bound_repo_root"] and repo_root is None:
        errors.append("work package binding bound_repo_root is required")
    if not _SHA256_RE.fullmatch(normalized["sha256"]):
        errors.append("work package binding sha256 is invalid")
    if normalized["sha256_kind"] not in {CANONICAL_HASH_KIND, CANONICAL_HASH_KIND_V1, LEGACY_HASH_KIND}:
        errors.append("work package binding sha256_kind is unsupported")
    for field in ("source_sha256", "document_sha256"):
        value = normalized.get(field)
        if value and not _SHA256_RE.fullmatch(str(value)):
            errors.append(f"work package binding {field} is invalid")
    if normalized.get("source_commit") and not re.fullmatch(r"[0-9a-f]{40,64}", str(normalized["source_commit"])):
        errors.append("work package binding source_commit is invalid")
    if normalized.get("contract_receipt_sha256") and not _SHA256_RE.fullmatch(str(normalized["contract_receipt_sha256"])):
        errors.append("work package binding contract_receipt_sha256 is invalid")
    locator = str(normalized.get("contract_receipt_locator") or "")
    if locator:
        locator_path = PurePosixPath(locator)
        if locator_path.is_absolute() or any(part in {"", ".", ".."} for part in locator_path.parts):
            errors.append("work package binding contract_receipt_locator must be state-root-relative")
    missing_required = sorted(field for field in WORK_PACKAGE_BINDING_REQUIRED_FIELDS if field not in binding)
    if normalized["sha256_kind"] == CANONICAL_HASH_KIND and missing_required:
        (errors if strict else warnings).append(
            "canonical Work Package binding is missing required provenance fields: " + ", ".join(missing_required)
        )
    if normalized["sha256_kind"] == CANONICAL_HASH_KIND:
        for field in ("source_sha256", "document_sha256", "source_commit"):
            if not normalized.get(field):
                (errors if strict else warnings).append(f"canonical Work Package binding {field} is required")
    if expected_work_id and normalized["work_id"] != str(expected_work_id).strip().upper():
        errors.append("work package binding belongs to another work item")
    if normalized["recommended_authority"] not in AUTHORITY_ROLES:
        errors.append("work package binding recommended_authority is invalid")
    if normalized["uncertainty_class"] not in UNCERTAINTY_CLASSES:
        errors.append("work package binding uncertainty_class is invalid")
    path: Path | None = None
    package: dict[str, Any] | None = None
    if not errors:
        try:
            path = resolve_work_package_binding(normalized, repo_root=repo_root)
            if not path.is_file():
                if strict:
                    errors.append(f"bound work package no longer exists: {path}")
                else:
                    package, receipt_errors = _load_contract_receipt(normalized, state_root=state_root)
                    if receipt_errors:
                        errors.extend([f"bound work package no longer exists: {path}", *receipt_errors])
                    else:
                        warnings.append(
                            "live Work Package path is unavailable; historical verification used the immutable contract receipt"
                        )
            else:
                raw_digest = file_sha256(path)
                package = load_work_package(path, expected_work_id=expected_work_id or normalized["work_id"])
                root = repo_root or normalized["bound_repo_root"]
                package = normalize_package_file_fences(package, repo_root=root)
                if normalized["sha256_kind"] == LEGACY_HASH_KIND:
                    if raw_digest != normalized["sha256"]:
                        errors.append(
                            f"bound work package hash changed: expected {normalized['sha256']}, observed {raw_digest}"
                        )
                else:
                    contract_digest = work_package_contract_sha256(
                        package,
                        hash_kind=normalized["sha256_kind"],
                    )
                    if contract_digest != normalized["sha256"]:
                        errors.append(
                            "bound Work Package material contract changed: "
                            f"expected {normalized['sha256']}, observed {contract_digest}"
                        )
                    if normalized.get("source_sha256") and raw_digest != normalized["source_sha256"]:
                        (errors if strict else warnings).append(
                            "Work Package source bytes changed without changing the material contract; "
                            "review formatting/advisory edits and rebind when appropriate"
                        )
                    document_digest = work_package_document_sha256(package)
                    if normalized.get("document_sha256") and document_digest != normalized["document_sha256"]:
                        (errors if strict else warnings).append(
                            "Work Package non-material document fields changed while the material contract remained stable"
                        )
                    if strict and normalized["sha256_kind"] == CANONICAL_HASH_KIND:
                        provenance = _git_package_provenance(path, repo_root=root, required=True)
                        observed_commit = str(provenance.get("source_commit") or "")
                        source_commit = str(normalized.get("source_commit") or "")
                        if source_commit and observed_commit:
                            exists = _git(root, "cat-file", "-e", f"{source_commit}^{{commit}}")
                            if exists.returncode != 0:
                                errors.append(f"bound Work Package source_commit is unavailable: {source_commit}")
                            else:
                                shown = _git(root, "show", f"{source_commit}:{normalized['path']}")
                                if shown.returncode != 0:
                                    errors.append("bound Work Package path does not exist in source_commit")
                                else:
                                    import hashlib

                                    shown_digest = hashlib.sha256(shown.stdout.encode("utf-8")).hexdigest()
                                    if shown_digest != normalized.get("source_sha256"):
                                        errors.append("bound Work Package source_commit does not contain the bound source bytes")
                for field in ("package_version", "recommended_authority", "uncertainty_class"):
                    if package.get(field) != normalized.get(field):
                        errors.append(f"work package binding {field} does not match the document")
                if package.get("recommended_model") != normalized.get("recommended_model"):
                    warnings.append(
                        "Work Package recommended_model differs from the historical binding; model recommendations are advisory"
                    )
        except ActorError as exc:
            errors.append(str(exc))
            errors.extend(list(exc.problems or []))
            warnings.extend(list(exc.warnings or []))
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "package": package,
        "path": str(path) if path else None,
        "binding": normalized,
    }


def uncertainty_is_promotion(current: str, requested: str) -> bool:
    return UNCERTAINTY_RANK.get(str(requested).upper(), 0) > UNCERTAINTY_RANK.get(str(current).upper(), 0)


__all__ = [
    "AUTHORITY_ROLES",
    "CANONICAL_HASH_KIND",
    "CANONICAL_HASH_KIND_V1",
    "LEGACY_HASH_KIND",
    "VERIFICATION_EXECUTION_STRICT",
    "VERIFICATION_HISTORICAL_AUDIT",
    "UNCERTAINTY_CLASSES",
    "UNCERTAINTY_RANK",
    "WORK_PACKAGE_BINDING_SCHEMA_VERSION",
    "WORK_PACKAGE_CONTRACT_RECEIPT_SCHEMA_VERSION",
    "WORK_PACKAGE_MATERIAL_FIELDS",
    "WORK_PACKAGE_OPTIONAL_MATERIAL_FIELDS",
    "WORK_PACKAGE_SCHEMA_VERSION",
    "DEFAULT_EXECUTION_BUDGET",
    "DEFAULT_OUTPUT_POLICY",
    "DEFAULT_READ_CONTRACT",
    "archive_work_package_contract",
    "bind_work_package",
    "effective_execution_budget",
    "effective_output_policy",
    "effective_read_contract",
    "delivery_recorded_at",
    "load_work_package",
    "load_work_package_contract_receipt",
    "load_work_package_value",
    "normalize_package_file_fences",
    "normalize_package_path_entry",
    "normalize_work_package",
    "normalize_work_package_binding",
    "package_path_matches",
    "read_work_package_document",
    "resolve_work_package_binding",
    "resolve_delivery_lineage",
    "review_recorded_at",
    "uncertainty_is_promotion",
    "validate_changed_paths",
    "validate_work_package",
    "verify_work_package_binding",
    "work_package_contract_sha256",
    "work_package_document_sha256",
    "work_package_material_equal",
    "work_package_material_payload",
]
