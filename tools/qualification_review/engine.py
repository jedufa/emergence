"""Orchestration for the declarative qualification review specification."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import FormatChecker
from jsonschema.validators import validator_for

from tools.actors.canonical import CANONICAL_SHA256_KIND

from .io import InputStore, ReviewInputError, file_sha256, receipt_sha256
from .model import CheckResult, SemanticQuestion
from .validators import (
    validate_authority_transition,
    validate_expected_negative,
    validate_freshness,
    validate_hash_graph,
    validate_inventory,
    validate_json_assertion,
    validate_packet,
    validate_process_convergence,
)
from .validators.temporal import format_utc, parse_utc

PACKAGE_ROOT = Path(__file__).resolve().parent
SPEC_SCHEMA_PATH = PACKAGE_ROOT / "schemas" / "review_spec.v1.schema.json"
RECEIPT_SCHEMA_PATH = PACKAGE_ROOT / "schemas" / "review_receipt.v1.schema.json"
RISK_RULES_PATH = PACKAGE_ROOT / "rules" / "risk_triggers.v1.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _schema_errors(document: Any, schema: dict) -> list[str]:
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    errors = sorted(
        validator_cls(schema, format_checker=FormatChecker()).iter_errors(document),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    rendered = []
    for error in errors:
        location = "/" + "/".join(str(item) for item in error.absolute_path)
        rendered.append(f"{location}: {error.message}")
    return rendered


def _asset_hashes() -> list[dict[str, str]]:
    assets = [
        {"path": "package://rules/risk_triggers.v1.json", "sha256": file_sha256(RISK_RULES_PATH)},
        {"path": "package://schemas/review_spec.v1.schema.json", "sha256": file_sha256(SPEC_SCHEMA_PATH)},
        {"path": "package://schemas/review_receipt.v1.schema.json", "sha256": file_sha256(RECEIPT_SCHEMA_PATH)},
    ]
    implementation_paths = sorted(PACKAGE_ROOT.glob("*.py")) + sorted(
        (PACKAGE_ROOT / "validators").glob("*.py")
    )
    assets.extend(
        {
            "path": f"package://implementation/{path.relative_to(PACKAGE_ROOT).as_posix()}",
            "sha256": file_sha256(path),
        }
        for path in implementation_paths
    )
    return assets


def _declared_input_paths(spec: dict[str, Any], store: InputStore) -> set[str]:
    paths: set[str] = set()

    def add(value: str) -> None:
        paths.add(store.key(store.resolve(value)))

    for row in spec.get("packet_validations", []):
        add(row["path"])
        add(row["schema_path"])
    for row in spec.get("hash_graphs", []):
        for node in row["nodes"]:
            add(node["path"])
    for section in (
        "json_assertions",
        "freshness_checks",
        "expected_negative_receipts",
        "process_convergence",
    ):
        for row in spec.get(section, []):
            add(row["path"])
    for row in spec.get("authority_transitions", []):
        for precondition in row.get("preconditions", []):
            add(precondition["path"])
    for row in spec.get("inventories", []):
        root = store.resolve(row["root"], require_file=False)
        if not root.is_dir():
            raise ReviewInputError(f"inventory root is not a directory: {row['root']}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ReviewInputError(f"inventory contains symlink: {path}")
            if path.is_file():
                paths.add(store.key(path))
    return paths


def _counts(checks: list[CheckResult]) -> dict[str, int]:
    return {
        "checks": len(checks),
        "passed": sum(check.status == "PASS" for check in checks),
        "failed": sum(check.status == "FAIL" for check in checks),
        "warned": sum(check.status == "WARN" for check in checks),
        "errors": sum(len(check.errors) for check in checks),
        "warnings": sum(len(check.warnings) for check in checks),
    }


def _build_receipt(
    *,
    review_id: str,
    now: datetime,
    spec_sha256: str,
    checks: list[CheckResult],
    questions: list[SemanticQuestion],
    store: InputStore,
    reusable: bool,
    valid_until: datetime | None,
    reused: bool = False,
    source_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    counts = _counts(checks)
    status = "FAIL" if counts["failed"] or counts["errors"] else "PASS"
    question_rows = [
        question.to_dict()
        for question in sorted(
            set(questions), key=lambda row: (row.trigger_type, row.trigger_id, row.question)
        )
    ]
    receipt: dict[str, Any] = {
        "schema_version": "eiel_qualification_mechanical_review_receipt_v1",
        "review_id": review_id,
        "generated_at_utc": format_utc(now),
        "hash_contract": {
            "file_hash_kind": "sha256_raw_bytes_v1",
            "receipt_digest_kind": CANONICAL_SHA256_KIND,
        },
        "spec_sha256": spec_sha256,
        "rules": _asset_hashes(),
        "input_hashes": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(store.hashes.items())
        ],
        "MECHANICAL_REVIEW": {
            "status": status,
            "counts": counts,
            "checks": [check.to_dict() for check in checks],
            "errors": [
                f"{check.check_id}: {error}" for check in checks for error in check.errors
            ],
            "warnings": [
                f"{check.check_id}: {warning}"
                for check in checks
                for warning in check.warnings
            ],
        },
        "SEMANTIC_REVIEW_REQUIRED": {
            "status": "REQUIRED" if question_rows else "NOT_REQUIRED",
            "question_count": len(question_rows),
            "questions": question_rows,
        },
        "freshness": {
            "reusable": bool(reusable and status == "PASS" and valid_until is not None),
            "checked_at_utc": format_utc(now),
            "valid_until_utc": format_utc(valid_until) if valid_until else None,
        },
        "reuse": {
            "reused": reused,
            "source_receipt_sha256": source_receipt_sha256,
        },
    }
    receipt["receipt_sha256"] = receipt_sha256(receipt)
    errors = _schema_errors(receipt, _load_json(RECEIPT_SCHEMA_PATH))
    if errors:
        raise RuntimeError(f"generated invalid qualification review receipt: {errors}")
    return receipt


def _reuse_receipt(
    candidate_path: Path,
    *,
    spec_path: Path,
    spec_sha256: str,
    now: datetime,
) -> dict[str, Any] | None:
    try:
        candidate = _load_json(candidate_path)
        spec = _load_json(spec_path)
        if _schema_errors(spec, _load_json(SPEC_SCHEMA_PATH)):
            return None
        if _schema_errors(candidate, _load_json(RECEIPT_SCHEMA_PATH)):
            return None
        if candidate["receipt_sha256"] != receipt_sha256(candidate):
            return None
        if candidate["MECHANICAL_REVIEW"]["status"] != "PASS":
            return None
        if not candidate["freshness"]["reusable"]:
            return None
        if candidate["spec_sha256"] != spec_sha256:
            return None
        if candidate["rules"] != _asset_hashes():
            return None
        valid_until = parse_utc(candidate["freshness"]["valid_until_utc"])
        if now > valid_until:
            return None
        store = InputStore(spec_path.parent)
        input_paths = [row["path"] for row in candidate["input_hashes"]]
        if len(input_paths) != len(set(input_paths)):
            return None
        expected_paths = _declared_input_paths(spec, store)
        if set(input_paths) != expected_paths:
            return None
        candidate_hashes = {row["path"]: row["sha256"] for row in candidate["input_hashes"]}
        for path in sorted(expected_paths):
            if store.hash_file(store.resolve(path)) != candidate_hashes[path]:
                return None
        reused = dict(candidate)
        reused["generated_at_utc"] = format_utc(now)
        reused["freshness"] = dict(candidate["freshness"])
        reused["freshness"]["checked_at_utc"] = format_utc(now)
        reused["reuse"] = {
            "reused": True,
            "source_receipt_sha256": candidate["receipt_sha256"],
        }
        reused["receipt_sha256"] = receipt_sha256(reused)
        if _schema_errors(reused, _load_json(RECEIPT_SCHEMA_PATH)):
            return None
        return reused
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError, ReviewInputError):
        return None


def review_spec(
    spec_path: Path,
    *,
    now: datetime | None = None,
    reuse_receipt: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one JSON spec without executing commands or controlling processes."""

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    spec_path = spec_path.resolve()
    spec_digest = file_sha256(spec_path)
    try:
        spec = _load_json(spec_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        store = InputStore(spec_path.parent)
        check = CheckResult("review_spec.load", "review_spec_schema")
        check.fail(f"invalid JSON review spec: {exc}")
        return _build_receipt(
            review_id="INVALID_SPEC",
            now=now,
            spec_sha256=spec_digest,
            checks=[check],
            questions=[],
            store=store,
            reusable=False,
            valid_until=None,
        )
    if reuse_receipt is not None:
        reused = _reuse_receipt(
            reuse_receipt.resolve(), spec_path=spec_path, spec_sha256=spec_digest, now=now
        )
        if reused is not None:
            return reused

    store = InputStore(spec_path.parent)
    checks: list[CheckResult] = []
    questions: list[SemanticQuestion] = []
    schema_check = CheckResult("review_spec.schema", "review_spec_schema")
    for error in _schema_errors(spec, _load_json(SPEC_SCHEMA_PATH)):
        schema_check.fail(error)
    checks.append(schema_check)
    if schema_check.status == "FAIL":
        return _build_receipt(
            review_id=spec.get("review_id", "INVALID_SPEC") if isinstance(spec, dict) else "INVALID_SPEC",
            now=now,
            spec_sha256=spec_digest,
            checks=checks,
            questions=questions,
            store=store,
            reusable=False,
            valid_until=None,
        )

    id_check = CheckResult("review_spec.unique_check_ids", "review_spec_integrity")
    ids: list[str] = []
    for section in (
        "packet_validations", "hash_graphs", "json_assertions", "inventories",
        "freshness_checks", "authority_transitions", "expected_negative_receipts",
        "process_convergence",
    ):
        ids.extend(row["id"] for row in spec.get(section, []))
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        id_check.fail(f"check IDs must be globally unique: {duplicates}")
    checks.append(id_check)

    rules = _load_json(RISK_RULES_PATH)
    if rules.get("schema_version") != spec["rules_version"]:
        rules_check = CheckResult("review_spec.rules_version", "review_spec_integrity")
        rules_check.fail(
            f"rules version mismatch: spec={spec['rules_version']!r}, "
            f"asset={rules.get('schema_version')!r}"
        )
        checks.append(rules_check)
    for row in spec.get("packet_validations", []):
        checks.append(validate_packet(row, store))
    for row in spec.get("hash_graphs", []):
        checks.append(validate_hash_graph(row, store))
    for row in spec.get("json_assertions", []):
        checks.append(validate_json_assertion(row, store))
    for row in spec.get("inventories", []):
        checks.append(validate_inventory(row, store))

    valid_until_values: list[datetime] = []
    for row in spec.get("freshness_checks", []):
        check, valid_until = validate_freshness(row, store, now)
        checks.append(check)
        if valid_until is not None:
            valid_until_values.append(valid_until)

    for row in spec.get("authority_transitions", []):
        check, triggered = validate_authority_transition(row, store, rules)
        checks.append(check)
        questions.extend(triggered)
    for row in spec.get("expected_negative_receipts", []):
        checks.append(validate_expected_negative(row, store))
    for row in spec.get("process_convergence", []):
        checks.append(validate_process_convergence(row, store))

    freshness_declared = spec["freshness_contract"]["reusable"]
    if freshness_declared and not valid_until_values:
        freshness_check = CheckResult("freshness_contract", "freshness")
        freshness_check.fail("reusable receipts require at least one freshness check")
        checks.append(freshness_check)
    for check in checks:
        if check.warnings:
            questions.append(
                SemanticQuestion(
                    trigger_id=f"{check.check_id}:mechanical_warning",
                    trigger_type="mechanical_warning",
                    question=rules["question_templates"]["mechanical_warning"],
                    evidence_pointers=tuple(sorted(store.hashes)),
                )
            )
    return _build_receipt(
        review_id=spec["review_id"],
        now=now,
        spec_sha256=spec_digest,
        checks=checks,
        questions=questions,
        store=store,
        reusable=freshness_declared,
        valid_until=min(valid_until_values) if valid_until_values else None,
    )


__all__ = ["CANONICAL_SHA256_KIND", "review_spec"]
