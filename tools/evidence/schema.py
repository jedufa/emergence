"""Low-level schema and file-shape helpers for evidence checks."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

csv.field_size_limit(sys.maxsize)


def read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str], str | None]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader), list(reader.fieldnames or []), None
    except Exception as exc:
        return [], [], f"{type(exc).__name__}: {exc}"


def parse_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def non_finite_values(row: Mapping[str, Any], columns: Iterable[str]) -> list[str]:
    bad: list[str] = []
    for column in columns:
        if column not in row or row.get(column) in (None, ""):
            continue
        try:
            parsed = float(row[column])
        except (TypeError, ValueError):
            bad.append(column)
            continue
        if not math.isfinite(parsed):
            bad.append(column)
    return bad


def missing_columns(columns: Iterable[str], required: Iterable[str]) -> list[str]:
    present = set(columns)
    return sorted(column for column in required if column not in present)


def truthy_number(value: Any) -> bool:
    parsed = parse_number(value)
    if parsed is None:
        return str(value).strip().lower() in {"true", "yes", "active"}
    return parsed != 0.0


def json_field(config: Mapping[str, Any], section: str, field: str) -> Any:
    section_value = config.get(section)
    if not isinstance(section_value, Mapping):
        return None
    return section_value.get(field)


def values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, float):
        parsed = parse_number(actual)
        return parsed is not None and parsed == expected
    return actual == expected
