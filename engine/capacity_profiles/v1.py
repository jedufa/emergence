"""Strict, side-effect-free model for ``eiel_capacity_profile_v1`` data.

The loader does not read environment variables, inspect devices, create paths,
or apply the resulting policy. Consumers must opt in explicitly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "eiel_capacity_profile_v1"
_ID_RE = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*\Z")
_DEVICE_RE = re.compile(
    r"(?:cpu|auto|cuda:[0-9]+|uuid:[A-Za-z0-9][A-Za-z0-9._:+-]{0,127})\Z"
)
_TOP_KEYS = frozenset(
    {
        "schema_version",
        "profile_id",
        "description",
        "build",
        "paths",
        "validation",
        "campaign",
        "benchmark",
    }
)
_BUILD_KEYS = frozenset({"max_jobs"})
_PATH_KEYS = frozenset({"runtime_root", "artifact_root"})
_VALIDATION_KEYS = frozenset(
    {"max_parallel_steps", "capture_bytes", "failure_policy"}
)
_CAMPAIGN_KEYS = frozenset({"workers"})
_BENCHMARK_KEYS = frozenset(
    {
        "batch_cells",
        "warmups",
        "repetitions",
        "device_selector",
        "objective",
        "memory_headroom_bytes",
    }
)
_FAILURE_POLICIES = frozenset({"continue", "fail_fast"})
_OBJECTIVES = frozenset({"correctness", "latency", "throughput"})

DEFAULTS: dict[str, dict[str, Any]] = {
    "build": {"max_jobs": 1},
    "validation": {
        "max_parallel_steps": 1,
        "capture_bytes": 1_048_576,
        "failure_policy": "fail_fast",
    },
    "campaign": {"workers": 1},
    "benchmark": {
        "batch_cells": 1,
        "warmups": 0,
        "repetitions": 1,
        "device_selector": "cpu",
        "objective": "correctness",
        "memory_headroom_bytes": 536_870_912,
    },
}


class CapacityProfileError(RuntimeError):
    """A capacity profile does not satisfy the v1 contract."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _unique_yaml_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise CapacityProfileError(f"duplicate YAML field {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_yaml_mapping,
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CapacityProfileError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise CapacityProfileError(f"non-finite JSON number {value!r} is not allowed")


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CapacityProfileError(f"{name} must be an object with string keys")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], allowed: frozenset[str], *, name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise CapacityProfileError(f"{name} contains unsupported fields: {unknown}")


def _identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise CapacityProfileError(f"{name} must be a lowercase stable identifier")
    return value


def _integer(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise CapacityProfileError(
            f"{name} must be an integer from {minimum} through {maximum}"
        )
    return value


def _choice(value: Any, choices: frozenset[str], *, name: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise CapacityProfileError(f"{name} must be one of {sorted(choices)}")
    return value


def _absolute_path(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CapacityProfileError(f"{name} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise CapacityProfileError(f"{name} must be an absolute normalized path")
    return path.as_posix()


def _section(
    profile: Mapping[str, Any],
    key: str,
    allowed: frozenset[str],
) -> dict[str, Any]:
    section = _object(profile.get(key, {}), name=key)
    _exact_keys(section, allowed, name=key)
    return section


def validate_profile(value: Any) -> dict[str, Any]:
    """Return a fully populated canonical profile without applying it."""

    profile = _object(value, name="profile")
    _exact_keys(profile, _TOP_KEYS, name="profile")
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise CapacityProfileError(f"schema_version must be {SCHEMA_VERSION!r}")
    profile_id = _identifier(profile.get("profile_id"), name="profile_id")
    description = profile.get("description", "")
    if not isinstance(description, str):
        raise CapacityProfileError("description must be a string")

    build = _section(profile, "build", _BUILD_KEYS)
    paths = _section(profile, "paths", _PATH_KEYS)
    validation = _section(profile, "validation", _VALIDATION_KEYS)
    campaign = _section(profile, "campaign", _CAMPAIGN_KEYS)
    benchmark = _section(profile, "benchmark", _BENCHMARK_KEYS)

    runtime_root = _absolute_path(
        paths.get("runtime_root", "/tmp/eiel-runtime"), name="paths.runtime_root"
    )
    artifact_root = _absolute_path(
        paths.get("artifact_root", str(Path(runtime_root) / "artifacts")),
        name="paths.artifact_root",
    )
    device_selector = benchmark.get(
        "device_selector", DEFAULTS["benchmark"]["device_selector"]
    )
    if not isinstance(device_selector, str) or _DEVICE_RE.fullmatch(device_selector) is None:
        raise CapacityProfileError(
            "benchmark.device_selector must be 'cpu', 'auto', 'cuda:<index>', "
            "or 'uuid:<device-uuid>'"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile_id,
        "description": description,
        "build": {
            "max_jobs": _integer(
                build.get("max_jobs", DEFAULTS["build"]["max_jobs"]),
                name="build.max_jobs",
                minimum=1,
                maximum=256,
            )
        },
        "paths": {
            "runtime_root": runtime_root,
            "artifact_root": artifact_root,
        },
        "validation": {
            "max_parallel_steps": _integer(
                validation.get(
                    "max_parallel_steps",
                    DEFAULTS["validation"]["max_parallel_steps"],
                ),
                name="validation.max_parallel_steps",
                minimum=1,
                maximum=256,
            ),
            "capture_bytes": _integer(
                validation.get(
                    "capture_bytes", DEFAULTS["validation"]["capture_bytes"]
                ),
                name="validation.capture_bytes",
                minimum=0,
                maximum=1_073_741_824,
            ),
            "failure_policy": _choice(
                validation.get(
                    "failure_policy", DEFAULTS["validation"]["failure_policy"]
                ),
                _FAILURE_POLICIES,
                name="validation.failure_policy",
            ),
        },
        "campaign": {
            "workers": _integer(
                campaign.get("workers", DEFAULTS["campaign"]["workers"]),
                name="campaign.workers",
                minimum=1,
                maximum=256,
            )
        },
        "benchmark": {
            "batch_cells": _integer(
                benchmark.get(
                    "batch_cells", DEFAULTS["benchmark"]["batch_cells"]
                ),
                name="benchmark.batch_cells",
                minimum=1,
                maximum=65_536,
            ),
            "warmups": _integer(
                benchmark.get("warmups", DEFAULTS["benchmark"]["warmups"]),
                name="benchmark.warmups",
                minimum=0,
                maximum=1_000,
            ),
            "repetitions": _integer(
                benchmark.get(
                    "repetitions", DEFAULTS["benchmark"]["repetitions"]
                ),
                name="benchmark.repetitions",
                minimum=1,
                maximum=10_000,
            ),
            "device_selector": device_selector,
            "objective": _choice(
                benchmark.get(
                    "objective", DEFAULTS["benchmark"]["objective"]
                ),
                _OBJECTIVES,
                name="benchmark.objective",
            ),
            "memory_headroom_bytes": _integer(
                benchmark.get(
                    "memory_headroom_bytes",
                    DEFAULTS["benchmark"]["memory_headroom_bytes"],
                ),
                name="benchmark.memory_headroom_bytes",
                minimum=0,
                maximum=9_223_372_036_854_775_807,
            ),
        },
    }


def load_profile(path: str | Path) -> dict[str, Any]:
    """Load strict JSON or YAML and return its canonical representation."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise CapacityProfileError(f"cannot read profile: {source}") from exc
    try:
        if source.suffix.lower() == ".json":
            value = json.loads(
                text,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        elif source.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.load(text, Loader=_UniqueKeySafeLoader)
        else:
            raise CapacityProfileError(
                "profile filename must end in .json, .yaml, or .yml"
            )
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CapacityProfileError(
            f"profile is not valid {source.suffix.lower()[1:].upper()}"
        ) from exc
    return validate_profile(value)
