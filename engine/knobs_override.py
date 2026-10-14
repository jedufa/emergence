from __future__ import annotations

"""
engine/knobs_override.py

Phase 1.5 — Knob Override Layer

This module provides a single, deterministic mechanism to override experiment knobs via:
  --knob section.field=value   (repeatable)

Key properties:
- Applied exactly once in main.py after normal CLI flags, before run_dir creation and RNG/derived config.
- Fail-fast validation: unknown keys or type mismatches abort the run.
- Type parsing uses dataclass field annotations (no guessing).
- Genome overrides are stored as a template map (field -> raw string) and applied to each Genome instance at spawn time.

Supported types (v1):
- bool, int, float, str
- Optional[T] where T is one of the above

Supported types (v1.1 fix):
- dict / Dict[...] where the override is either:
    * inline JSON object: {"k": 1}
    * file pointer: @path/to/file.json   (decoded JSON must be an object/dict)
"""

from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Tuple, get_args, get_origin, Union
import json
import os
import difflib

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


# ---------------------------------------------------------------------------
# Deprecated knob aliases
# ---------------------------------------------------------------------------
#
# Experiment registries tend to evolve faster than config schemas.
# We keep strict validation, but allow a small set of explicit, auditable
# aliases for renamed knobs so you don't burn hours on avoidable KeyErrors.
#
# Rules:
# - Only map names that were previously used in manifests/registries.
# - Keep the mapping tiny and intentional.
# - Canonical names are the destination keys.

DEPRECATED_SIM_CFG_FIELD_ALIASES: Dict[str, str] = {
    # Chaos Week 2 (older registry naming): "late life pressure" -> Option C late-life selection pressure.
    "late_life_pressure_enable": "late_pred_cost_enable",
    "late_life_pressure_weight": "late_pred_cost_weight",

    # Chaos Week 2 (older registry naming): "developmental lock-in" -> identity drift capability tax.
    # NOTE: This is the closest existing mechanic in baseline-120.
    "dev_lockin_enable": "identity_drift_enable",
    "dev_lockin_strength": "identity_drift_strength",
    "dev_lockin_scale": "identity_drift_scale",
}
# Behavior-bearing fields proven by the accepted WRC control-path audits to
# have no production mechanic consumer.  Keeping them generically assignable
# would advertise scientific control that does not exist.  They therefore fail
# closed at the generic override boundary until an actual mechanic is present.
DEAD_PRODUCTION_KNOB_FIELDS: Dict[str, frozenset[str]] = {
    "env_cfg": frozenset(
        {
            "patch_waste_fertility_effect_enable",
            "patch_waste_fertility_damage_rate",
            "patch_fertility_recovery_rate",
            "patch_fertility_recovery_target",
            "jungle_recovery_fertility_enable",
            "jungle_resource_spread_enable",
            "wdl_delayed_hazard_regrowth_suppression_enable",
            "prediction_demand_cycle_mode",
            "injury_carry_decay",
            "injury_carry_decay_requires_alive",
            "habacclim_observation_scalar_name",
        }
    ),
    "pop_cfg": frozenset(),
    "sim_cfg": frozenset(),
}



def _parse_bool(s: str) -> bool:
    """Parse boolean from accepted tokens.

    Accepted (case-insensitive):
      true/false/1/0/yes/no/on/off
    """
    v = str(s).strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"invalid bool token: {s!r}")


def _is_optional(tp: Any) -> Tuple[bool, Any]:
    """Return (is_optional, inner_type) for Optional[T]."""
    origin = get_origin(tp)
    if origin is Union:
        args = list(get_args(tp))
        if len(args) == 2 and type(None) in args:
            inner = args[0] if args[1] is type(None) else args[1]
            return True, inner
    return False, tp


def _resolve_at_path(raw_path: str) -> str:
    """Resolve an @path payload to an absolute filesystem path.

    Resolution rules (deterministic, minimal magic):
    - Expand ~
    - If already absolute: keep
    - Else: resolve relative to current working directory
      (the harness runs with repo root as cwd in standard operation).
    """
    p = os.path.normpath(os.path.expanduser(raw_path))
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(os.getcwd(), p))


def _parse_value_for_type(tp: Any, raw: str) -> Any:
    """Parse a string value according to a dataclass field annotation.

    Supported:
      - bool, int, float, str
      - Optional[bool/int/float/str] (empty string or 'null' -> None)
      - dict / Dict[...] :
          * '@path/to/file.json' -> load JSON file and return object (must be a dict)
          * inline JSON like '{"k": 1}' -> json.loads (must decode to dict)
    """
    is_opt, inner = _is_optional(tp)
    s = str(raw).strip()

    if is_opt and (s == "" or s.lower() == "null"):
        return None

    t = inner

    # Allow typing aliases like 'float' or 'builtins.float' (handled upstream for strings)
    if t is bool:
        return _parse_bool(s)
    if t is int:
        return int(s)
    if t is float:
        return float(s)
    if t is str:
        return s

    # Support dict-typed fields (common for structured env assets).
    origin = get_origin(t)
    if t is dict or origin is dict:
        # File pointer form: @relative/or/absolute/path.json
        if s.startswith("@"):
            path = s[1:].strip()
            if not path:
                raise ValueError(f"invalid dict override: empty @path (raw={raw!r})")
            fs_path = _resolve_at_path(path)
            if not os.path.exists(fs_path):
                raise FileNotFoundError(
                    f"dict override file not found: {fs_path!r} (raw={raw!r})"
                )
            with open(fs_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                raise TypeError(
                    f"dict override JSON must decode to dict, got {type(obj)} from {fs_path!r}"
                )
            return obj

        # Inline JSON object
        try:
            obj = json.loads(s)
        except Exception as e:
            raise ValueError(f"invalid dict override JSON: {s!r}") from e
        if not isinstance(obj, dict):
            raise TypeError(
                f"dict override must be a JSON object (dict), got {type(obj)} (raw={raw!r})"
            )
        return obj

    raise TypeError(f"unsupported knob type: {tp!r} (raw={raw!r})")


def _canon_value(v: Any) -> str:
    """Canonical string form for audit trails."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, dict):
        # Stable JSON for audit trails (sorted keys, no whitespace variance).
        return json.dumps(v, sort_keys=True, separators=(",", ":"))
    return str(v)


def _dataclass_field_map(obj: Any) -> Dict[str, Any]:
    """Return field_name -> field_type map for a dataclass instance.

    Important:
        In baseline-105 we run with ``from __future__ import annotations`` in several
        modules. In that mode, dataclass field annotations can be stored as strings
        (e.g. "float") which breaks type-based parsing.

        To stay deterministic while supporting postponed evaluation, we resolve
        annotations using ``typing.get_type_hints`` on the dataclass class, falling
        back to raw field types when resolution fails.

    Args:
        obj: A dataclass instance.

    Returns:
        Dict[str, Any]: Mapping from field names to resolved annotation types.
    """
    if not is_dataclass(obj):
        raise TypeError(f"target is not a dataclass: {type(obj)}")

    cls = obj.__class__

    # Best-effort resolution of postponed annotations.
    try:
        import sys as _sys
        import typing as _typing

        mod = _sys.modules.get(cls.__module__)
        globalns = vars(mod) if mod is not None else {}
        hints = _typing.get_type_hints(cls, globalns=globalns, localns=dict(vars(cls)))
    except Exception:
        hints = {}

    out: Dict[str, Any] = {}
    for f in fields(obj):
        tp = hints.get(f.name, f.type)

        # If the resolved type is still a string, handle the common primitives.
        if isinstance(tp, str):
            s = tp.strip()
            if s in ("float", "builtins.float"):
                tp = float
            elif s in ("int", "builtins.int"):
                tp = int
            elif s in ("bool", "builtins.bool"):
                tp = bool
            elif s in ("str", "builtins.str"):
                tp = str
            elif s in ("dict", "builtins.dict"):
                tp = dict

        out[f.name] = tp

    return out


def _apply_to_dataclass(obj: Any, updates: Dict[str, str]) -> List[str]:
    """Apply field updates to a dataclass instance using its annotations.

    Args:
        obj: dataclass instance
        updates: mapping field_name -> raw string value

    Returns:
        List of canonical 'section.field=value' strings is handled by caller.
        Here we return canonical 'field=value' strings.
    """
    fmap = _dataclass_field_map(obj)
    applied: List[str] = []
    for field_name, raw in updates.items():
        if field_name not in fmap:
            # Give a useful error (strict, but not sadistic).
            choices = sorted(fmap.keys())
            near = difflib.get_close_matches(field_name, choices, n=5, cutoff=0.55)
            hint = f". Did you mean: {', '.join(near)}" if near else ""
            raise KeyError(
                f"unknown field: {field_name}{hint}. Available fields: {', '.join(choices)}"
            )
        parsed = _parse_value_for_type(fmap[field_name], raw)
        setattr(obj, field_name, parsed)
        applied.append(f"{field_name}={_canon_value(parsed)}")
    return applied


def parse_knob_pairs(knob_pairs: List[str]) -> List[Tuple[str, str, str]]:
    """Parse --knob pairs into (section, field, raw_value).

    Expected format: section.field=value
    """
    out: List[Tuple[str, str, str]] = []
    for item in knob_pairs or []:
        s = str(item).strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"KNOB_INVALID: expected section.field=value, got {item!r}")
        key, raw = s.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if "." not in key:
            raise ValueError(f"KNOB_INVALID: expected section.field=value, got {item!r}")
        section, field = key.split(".", 1)
        section = section.strip()
        field = field.strip()
        if not section or not field:
            raise ValueError(f"KNOB_INVALID: expected section.field=value, got {item!r}")
        out.append((section, field, raw))
    return out


def apply_knob_overrides(
    *,
    knob_pairs: List[str],
    env_cfg: Any,
    pop_cfg: Any,
    sim_cfg: Any,
) -> Tuple[List[str], Dict[str, str]]:
    """Apply --knob overrides to env_cfg/pop_cfg/sim_cfg, and collect genome overrides.

    Precedence:
      code defaults < explicit CLI flags < --knob overrides

    Genome:
      CLI section 'genome' is mapped to runtime Genome instances at spawn time.
      We do NOT apply genome overrides here; we only validate and store them.

    Returns:
        applied_overrides: list of canonical 'section.field=value' strings
        genome_overrides: dict of Genome field -> raw string (for later parsing at spawn time)
    """
    parsed = parse_knob_pairs(knob_pairs)

    allowed = {
        "env_cfg": env_cfg,
        "pop_cfg": pop_cfg,
        "sim_cfg": sim_cfg,
        # "genome" handled separately
    }

    # collect per-target updates
    updates_env: Dict[str, str] = {}
    updates_pop: Dict[str, str] = {}
    updates_sim: Dict[str, str] = {}
    genome_overrides: Dict[str, str] = {}

    for section, field, raw in parsed:
        if section == "genome":
            # validate existence later at spawn-time parse; here we only store template
            genome_overrides[field] = raw
            continue
        if section not in allowed:
            raise ValueError(f"INVALID_KNOB: unknown section {section!r}")
        if field in DEAD_PRODUCTION_KNOB_FIELDS.get(section, frozenset()):
            raise ValueError(
                "INVALID_KNOB: production control has no runtime consumer: "
                f"{section}.{field}"
            )
        if section == "env_cfg":
            updates_env[field] = raw
        elif section == "pop_cfg":
            updates_pop[field] = raw
        elif section == "sim_cfg":
            # Backwards-compatible aliases for renamed knobs.
            canonical_field = DEPRECATED_SIM_CFG_FIELD_ALIASES.get(field, field)
            updates_sim[canonical_field] = raw

    applied: List[str] = []
    if updates_env:
        a = _apply_to_dataclass(env_cfg, updates_env)
        applied.extend([f"env_cfg.{x}" for x in a])
    if updates_pop:
        a = _apply_to_dataclass(pop_cfg, updates_pop)
        applied.extend([f"pop_cfg.{x}" for x in a])
    if updates_sim:
        a = _apply_to_dataclass(sim_cfg, updates_sim)
        applied.extend([f"sim_cfg.{x}" for x in a])

    # genome overrides are stored as raw; canonicalize audit string by parsing via Genome type when applied.
    # For now, include raw strings as-is in audit list (still deterministic).
    for k, v in genome_overrides.items():
        applied.append(f"genome.{k}={v}")

    return applied, genome_overrides


def verify_knob_override_readback(
    *,
    applied_overrides: List[str],
    env_cfg: Any,
    pop_cfg: Any,
    sim_cfg: Any,
) -> List[Dict[str, Any]]:
    """Fail closed when a later profile or context overwrote an accepted knob.

    This is deliberately evaluated after all post-knob configuration layers in
    ``main.py``.  It turns the generic surface from membership-only assignment
    into requested/applied/readback proof for the final Python runtime config.
    Genome templates are verified at spawn time and are therefore omitted here.
    """

    targets = {"env_cfg": env_cfg, "pop_cfg": pop_cfg, "sim_cfg": sim_cfg}
    receipts: List[Dict[str, Any]] = []
    for item in applied_overrides:
        key, raw = str(item).split("=", 1)
        section, field_name = key.split(".", 1)
        if section == "genome":
            continue
        target = targets.get(section)
        if target is None:
            raise ValueError(f"INVALID_KNOB: unknown readback section {section!r}")
        field_map = _dataclass_field_map(target)
        if field_name not in field_map:
            raise ValueError(
                f"INVALID_KNOB: readback field disappeared: {section}.{field_name}"
            )
        requested = _parse_value_for_type(field_map[field_name], raw)
        readback = getattr(target, field_name)
        if type(readback) is not type(requested) or readback != requested:
            raise ValueError(
                "KNOB_OVERRIDDEN: accepted control differs after final config "
                f"assembly: {section}.{field_name} requested={requested!r} "
                f"readback={readback!r}"
            )
        receipts.append(
            {
                "schema_version": "eiel_generic_knob_readback_v1",
                "control": f"{section}.{field_name}",
                "requested": requested,
                "readback": readback,
                "status": "FINAL_CONFIG_READBACK_COMPLETE",
                "mechanic_connectivity_qualified": False,
                "wrc_admitted": False,
                "qualification_note": (
                    "generic knob readback proves final dataclass state only; "
                    "WRC requires a typed production-route control receipt"
                ),
            }
        )
    return receipts


def apply_genome_overrides_to_instance(genome: Any, overrides: Dict[str, str]) -> None:
    """Apply stored genome.* overrides to a Genome instance (best-effort).

    - Uses Genome dataclass annotations for type parsing.
    - Re-runs __post_init__ if present to re-clamp derived invariants.

    Args:
        genome: Genome instance
        overrides: dict[field -> raw string]
    """
    if not overrides:
        return
    if not is_dataclass(genome):
        # Genome should be a dataclass; if not, fail fast.
        raise TypeError(f"Genome is not a dataclass: {type(genome)}")

    fmap = _dataclass_field_map(genome)
    for field_name, raw in overrides.items():
        if field_name not in fmap:
            raise KeyError(f"INVALID_KNOB: genome.{field_name} does not exist")
        parsed = _parse_value_for_type(fmap[field_name], raw)
        setattr(genome, field_name, parsed)

    # Re-run post-init to enforce invariants if available.
    post = getattr(genome, "__post_init__", None)
    if callable(post):
        post()
