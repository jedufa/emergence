"""Evidence manifest loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REQUIRED_FIELDS: tuple[str, ...] = (
    "campaign_id",
    "profile",
    "git_commit",
    "branch",
    "worktree",
    "run_root",
    "registry_path",
    "scenario_matrix",
    "seed_list",
    "generations",
    "steps_per_generation",
    "population_size",
    "grid_size",
    "device",
    "planned_run_count",
    "completed_run_count",
    "failed_run_count",
    "config_files",
    "output_paths",
    "metrics_glob",
    "config_glob",
    "harness_manifest_paths",
    "harness_summary_paths",
    "timestamp_utc",
)

PATH_FIELDS: tuple[str, ...] = (
    "worktree",
    "run_root",
    "registry_path",
)

PATH_LIST_FIELDS: tuple[str, ...] = (
    "config_files",
    "output_paths",
    "harness_manifest_paths",
    "harness_summary_paths",
)

STRING_FIELDS = {
    "campaign_id",
    "profile",
    "git_commit",
    "branch",
    "worktree",
    "run_root",
    "registry_path",
    "device",
    "metrics_glob",
    "config_glob",
    "timestamp_utc",
}

INTEGER_FIELDS = {
    "generations",
    "steps_per_generation",
    "population_size",
    "grid_size",
    "planned_run_count",
    "completed_run_count",
    "failed_run_count",
}


class ManifestError(ValueError):
    """Raised when an evidence manifest is missing required structure."""

    def __init__(self, path: Path, errors: list[str]) -> None:
        self.path = path
        self.errors = errors
        super().__init__(f"{path}: " + "; ".join(errors))

    def as_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "errors": list(self.errors)}


@dataclass(frozen=True)
class EvidenceManifest:
    path: Path
    data: dict[str, Any]

    def to_json_text(self) -> str:
        return dumps_manifest(self.data)


def _as_abs(base: Path, value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def _normalize_paths(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    base = path.parent.resolve()
    normalized = dict(data)
    for field in PATH_FIELDS:
        if field in normalized:
            normalized[field] = _as_abs(base, normalized[field])
    for field in PATH_LIST_FIELDS:
        values = normalized.get(field)
        if isinstance(values, list):
            normalized[field] = [_as_abs(base, item) for item in values]
    return normalized


def _scenario_dimensions(matrix: Any) -> list[dict[str, Any]]:
    if isinstance(matrix, Mapping):
        dims = matrix.get("dimensions")
        if isinstance(dims, list):
            return [dict(item) for item in dims if isinstance(item, Mapping)]
        return [
            {"name": str(name), "values": values}
            for name, values in matrix.items()
            if name != "dimensions"
        ]
    return []


def scenario_dimensions(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _scenario_dimensions(manifest.get("scenario_matrix"))


def scenario_dimension_values(manifest: Mapping[str, Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for dim in scenario_dimensions(manifest):
        name = dim.get("name")
        values = dim.get("values")
        if isinstance(name, str) and isinstance(values, list):
            out[name] = list(values)
    return out


def validate_manifest_data(data: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"missing required field: {field}")
    for field in sorted(STRING_FIELDS.intersection(data.keys())):
        if not isinstance(data[field], str) or not str(data[field]).strip():
            errors.append(f"{field} must be a non-empty string")
    for field in sorted(INTEGER_FIELDS.intersection(data.keys())):
        if not isinstance(data[field], int) or isinstance(data[field], bool) or data[field] < 0:
            errors.append(f"{field} must be a non-negative integer")
    for field in sorted(PATH_LIST_FIELDS):
        if field in data and not isinstance(data[field], list):
            errors.append(f"{field} must be a list")
        elif field in data and not all(isinstance(item, str) for item in data[field]):
            errors.append(f"{field} must contain only strings")
    seeds = data.get("seed_list")
    if not isinstance(seeds, list) or not seeds:
        errors.append("seed_list must be a non-empty list")
    elif not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        errors.append("seed_list must contain integer seeds")

    dims = _scenario_dimensions(data.get("scenario_matrix"))
    if not dims:
        errors.append("scenario_matrix must define dimensions")
    for dim in dims:
        name = dim.get("name")
        values = dim.get("values")
        if not isinstance(name, str) or not name:
            errors.append("scenario_matrix dimensions must have non-empty string names")
        if not isinstance(values, list) or not values:
            errors.append(f"scenario_matrix dimension {name!r} must have non-empty values")
        elif not all(isinstance(value, (str, int)) for value in values):
            errors.append(f"scenario_matrix dimension {name!r} values must be strings or integers")
    return errors


def load_manifest(path: str | Path) -> EvidenceManifest:
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(manifest_path, [f"invalid JSON: {exc.msg} at line {exc.lineno}"]) from exc
    if not isinstance(raw, dict):
        raise ManifestError(manifest_path, ["manifest root must be a JSON object"])
    errors = validate_manifest_data(raw)
    if errors:
        raise ManifestError(manifest_path, errors)
    return EvidenceManifest(path=manifest_path, data=_normalize_paths(manifest_path, dict(raw)))


def dumps_manifest(data: Mapping[str, Any]) -> str:
    return json.dumps(dict(data), indent=2, sort_keys=True) + "\n"


def write_manifest(path: str | Path, data: Mapping[str, Any]) -> None:
    Path(path).write_text(dumps_manifest(data), encoding="utf-8")
