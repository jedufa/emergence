"""experiments.harness.schema

Registry validation and normalization for the Experiment Registry & Validation Harness (v1).

Goals:
- Fail fast on malformed registries.
- Enforce deterministic expansion rules.
- Keep the harness runner boring and predictable.

This module intentionally avoids fancy abstractions and keeps validation explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RegistryAxis:
    """A single registry axis definition.

    Attributes:
        name: Axis name (e.g. "dev_coupling").
        values: Ordered list of value labels (e.g. ["off", "on"]).
        flags: Mapping from value label -> list of CLI tokens.
    """

    name: str
    values: List[str]
    flags: Dict[str, List[str]]


@dataclass(frozen=True)
class RegistryNaming:
    """Naming rules.

    Attributes:
        run_tag_prefix: Prefix used in run_tag_format.
        run_id_format: Python format string used to generate run_id.
        run_tag_format: Python format string used to generate run_tag.
    """

    run_tag_prefix: str
    run_id_format: str
    run_tag_format: str


@dataclass(frozen=True)
class ExperimentRegistry:
    """Normalized experiment registry representation."""

    harness_api: int
    version: int
    name: str
    description: str
    intent: str
    matrix_order: List[str]
    axes: List[RegistryAxis]
    seeds: List[int]
    cli_defaults: List[List[str]]
    expected_artifacts: List[str]
    naming: RegistryNaming


def _require(d: Mapping[str, Any], key: str, ctx: str) -> Any:
    """Require a key in a mapping and return it.

    Args:
        d: Mapping to read.
        key: Required key.
        ctx: Human context for errors.

    Raises:
        ValueError: If the key is missing.
    """
    if key not in d:
        raise ValueError(f"Missing required field '{key}' in {ctx}.")
    return d[key]


def _as_str_list(x: Any, ctx: str) -> List[str]:
    """Coerce a value into a list of strings."""
    if x is None:
        return []
    if not isinstance(x, list):
        raise ValueError(f"Expected list in {ctx}, got {type(x).__name__}.")
    out: List[str] = []
    for i, v in enumerate(x):
        if not isinstance(v, (str, int, float)):
            raise ValueError(f"Expected scalar in {ctx}[{i}], got {type(v).__name__}.")
        out.append(str(v))
    return out


def _as_list_of_list_str(x: Any, ctx: str) -> List[List[str]]:
    """Coerce into list[list[str]] (for cli_defaults)."""
    if not isinstance(x, list):
        raise ValueError(f"Expected list in {ctx}, got {type(x).__name__}.")
    out: List[List[str]] = []
    for i, row in enumerate(x):
        if not isinstance(row, list):
            raise ValueError(f"Expected list in {ctx}[{i}], got {type(row).__name__}.")
        out.append(_as_str_list(row, f"{ctx}[{i}]") )
    return out


def _validate_seeds(seeds: Any) -> List[int]:
    """Validate seed hygiene: unique integers."""
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("'seeds' must be a non-empty list of integers.")
    out: List[int] = []
    for i, s in enumerate(seeds):
        if isinstance(s, bool):
            raise ValueError(f"Seed at index {i} must be int, got bool.")
        try:
            si = int(s)
        except Exception:
            raise ValueError(f"Seed at index {i} is not an int: {s!r}")
        out.append(si)
    if len(set(out)) != len(out):
        raise ValueError(f"Seeds must be unique. Got: {out}")
    return out


def load_and_validate_registry(doc: Mapping[str, Any]) -> ExperimentRegistry:
    """Validate and normalize a parsed registry document.

    Args:
        doc: Parsed YAML document (dict-like).

    Returns:
        ExperimentRegistry: Normalized registry.

    Raises:
        ValueError: If the registry is invalid.
    """
    ctx = "registry"

    harness_api = int(_require(doc, "harness_api", ctx))
    version = int(_require(doc, "version", ctx))
    name = str(_require(doc, "name", ctx))
    description = str(doc.get("description", "") or "")
    intent = str((doc.get("intent") or doc.get("question") or "") or "")

    matrix_order = _as_str_list(_require(doc, "matrix_order", ctx), "registry.matrix_order")
    if not matrix_order:
        raise ValueError("matrix_order must be a non-empty list.")
    if matrix_order[-1] != "seed":
        raise ValueError("matrix_order must end with 'seed' to guarantee stable ordering.")

    axes_block = _require(doc, "axes", ctx)
    if not isinstance(axes_block, dict) or not axes_block:
        raise ValueError("axes must be a non-empty mapping.")

    axes: List[RegistryAxis] = []
    for axis_name, axis_def in axes_block.items():
        if not isinstance(axis_def, dict):
            raise ValueError(f"Axis '{axis_name}' must be a mapping.")
        values = _as_str_list(_require(axis_def, "values", f"axis:{axis_name}"), f"axes.{axis_name}.values")
        flags_map_raw = _require(axis_def, "flags", f"axis:{axis_name}")
        if not isinstance(flags_map_raw, dict):
            raise ValueError(f"axes.{axis_name}.flags must be a mapping.")
        flags_map: Dict[str, List[str]] = {}
        for v in values:
            if v not in flags_map_raw:
                raise ValueError(f"axes.{axis_name}.flags missing entry for value '{v}'.")
            flags_map[v] = _as_str_list(flags_map_raw[v], f"axes.{axis_name}.flags.{v}")
        axes.append(RegistryAxis(name=str(axis_name), values=values, flags=flags_map))

    # matrix_order must reference all axes exactly once (plus seed)
    axis_names = [a.name for a in axes]
    axis_names_set = set(axis_names)
    mo_no_seed = [x for x in matrix_order if x != "seed"]
    if set(mo_no_seed) != axis_names_set:
        raise ValueError(
            f"matrix_order (excluding seed) must match axes keys. axes={axis_names} matrix_order={matrix_order}"
        )

    seeds = _validate_seeds(_require(doc, "seeds", ctx))

    cli_defaults = _as_list_of_list_str(_require(doc, "cli_defaults", ctx), "registry.cli_defaults")
    if not cli_defaults:
        raise ValueError("cli_defaults must be a non-empty list of CLI token lists.")

    expected_artifacts = _as_str_list(_require(doc, "expected_artifacts", ctx), "registry.expected_artifacts")
    if not expected_artifacts:
        raise ValueError("expected_artifacts must be a non-empty list.")
    if "manifest.json" not in expected_artifacts:
        raise ValueError("expected_artifacts must include 'manifest.json' (mandatory in this project).")

    naming_raw = _require(doc, "naming", ctx)
    if not isinstance(naming_raw, dict):
        raise ValueError("naming must be a mapping.")
    run_tag_prefix = str(_require(naming_raw, "run_tag_prefix", "naming"))
    run_id_format = str(_require(naming_raw, "run_id_format", "naming"))
    run_tag_format = str(_require(naming_raw, "run_tag_format", "naming"))
    naming = RegistryNaming(run_tag_prefix=run_tag_prefix, run_id_format=run_id_format, run_tag_format=run_tag_format)

    return ExperimentRegistry(
        harness_api=harness_api,
        version=version,
        name=name,
        description=description,
        intent=intent,
        matrix_order=matrix_order,
        axes=axes,
        seeds=seeds,
        cli_defaults=cli_defaults,
        expected_artifacts=expected_artifacts,
        naming=naming,
    )


# Preserve legacy persisted registry addresses; no separate state owner.
for _record in (RegistryAxis, RegistryNaming, ExperimentRegistry):
    _record.__module__ = "experiments.harness.schema"
