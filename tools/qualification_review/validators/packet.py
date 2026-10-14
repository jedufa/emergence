"""Packet and schema validation."""

from __future__ import annotations

from jsonschema import FormatChecker
from jsonschema.validators import validator_for

from ..io import InputStore, ReviewInputError
from ..model import CheckResult


def validate_packet(spec: dict, store: InputStore) -> CheckResult:
    result = CheckResult(spec["id"], "packet_schema")
    try:
        packet = store.read_json(spec["path"])
        schema = store.read_json(spec["schema_path"])
        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)
        errors = sorted(
            validator_cls(schema, format_checker=FormatChecker()).iter_errors(packet),
            key=lambda error: (list(error.absolute_path), error.message),
        )
        for error in errors:
            location = "/" + "/".join(str(item) for item in error.absolute_path)
            result.fail(f"{location}: {error.message}")
        result.details = {
            "packet": spec["path"],
            "schema": spec["schema_path"],
            "schema_error_count": len(errors),
        }
    except (ReviewInputError, ValueError) as exc:
        result.fail(str(exc))
    return result
