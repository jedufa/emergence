from __future__ import annotations

"""simulation/trait_injection_pkg/schema_v2.py

Trait Injection schema (v2) and deterministic validation helpers.

This module is dependency-light and safe to import from ControlPlane validation
or from runtime application logic.

Schema goals
------------
- Be strict enough to prevent malformed injections from entering the WAL/event log.
- Keep payload compact and deterministic (no timestamps, no wall-clock fields).

Authoritative apply phases (frozen per campaign)
----------------------------------------------
- post-reproduction-pre-selection
- post-selection-pre-mutation

Notes
-----
- Validation includes Genome locus tags to prevent silent no-ops.
- Duration is expressed in generations and must be >= 1.
"""

from dataclasses import fields
from typing import Any, Dict, Optional, Sequence, Tuple


TRAIT_INJECTION_SCHEMA_VERSION_V2 = "trait_injection_v2"

APPLY_PHASE_POST_REPRO_PRE_SELECTION = "post-reproduction-pre-selection"
APPLY_PHASE_POST_SELECTION_PRE_MUTATION = "post-selection-pre-mutation"

ALLOWED_APPLY_PHASES: Tuple[str, str] = (
    APPLY_PHASE_POST_REPRO_PRE_SELECTION,
    APPLY_PHASE_POST_SELECTION_PRE_MUTATION,
)


def _genome_field_names() -> Sequence[str]:
    """Return allowed Genome locus tags.

    Determinism note: dataclass field order is stable.
    """
    try:
        from agents.genome import Genome

        return tuple(f.name for f in fields(Genome))
    except Exception:
        # Fail closed: if Genome cannot be imported, do not allow any locus tags.
        return tuple()


_GENOME_FIELDS = set(_genome_field_names())


def normalize_duration_generations(spec: Dict[str, Any]) -> int:
    """Extract duration_generations from a trait injection spec.

    Accepted keys:
    - duration_generations (preferred)
    - duration (back-compat)
    """
    if not isinstance(spec, dict):
        return 0
    if "duration_generations" in spec:
        try:
            return int(spec.get("duration_generations"))
        except Exception:
            return 0
    if "duration" in spec:
        try:
            return int(spec.get("duration"))
        except Exception:
            return 0
    return 0


def validate_trait_injection_v2(
    spec: Any,
    *,
    frozen_apply_phase: Optional[str] = None,
) -> Tuple[bool, str]:
    """Validate a TraitInjection v2 spec.

    Parameters
    ----------
    spec:
        Candidate trait injection spec (expected dict).
    frozen_apply_phase:
        If provided, enforce that spec.apply_phase matches the campaign-frozen phase.

    Returns
    -------
    (ok, reason)
        ok=True if valid, else ok=False with deterministic reason.
    """
    if not isinstance(spec, dict):
        return False, "trait_injection must be a dict"

    sv = str(spec.get("schema_version", "") or "")
    if sv != TRAIT_INJECTION_SCHEMA_VERSION_V2:
        return False, "trait_injection.schema_version must be trait_injection_v2"

    ap = str(spec.get("apply_phase", "") or "")
    if ap not in ALLOWED_APPLY_PHASES:
        return False, "trait_injection.apply_phase not allowed"

    if frozen_apply_phase is not None and str(frozen_apply_phase) and ap != str(frozen_apply_phase):
        return False, "trait_injection.apply_phase mismatch vs campaign frozen phase"

    dur = normalize_duration_generations(spec)
    if dur < 1:
        return False, "trait_injection.duration_generations must be >= 1"

    locus_tags = spec.get("locus_tags", [])
    if not isinstance(locus_tags, list) or not locus_tags:
        return False, "trait_injection.locus_tags missing"

    for t in locus_tags:
        if not isinstance(t, str) or not t:
            return False, "trait_injection.locus_tags must be non-empty strings"
        if t not in _GENOME_FIELDS:
            return False, f"trait_injection.locus_tag unknown: {t}"

    allele_values = spec.get("allele_values", {})
    if not isinstance(allele_values, dict):
        return False, "trait_injection.allele_values must be a dict"

    # Require values for every locus tag (prevents silent partial injection).
    missing = [t for t in locus_tags if t not in allele_values]
    if missing:
        return False, "trait_injection.allele_values missing keys for locus_tags"

    # Target agent ids are optional; if provided, must be a list of ints.
    if "target_agent_ids" in spec and spec.get("target_agent_ids") is not None:
        ids = spec.get("target_agent_ids")
        if not isinstance(ids, list):
            return False, "trait_injection.target_agent_ids must be a list"
        for x in ids:
            try:
                int(x)
            except Exception:
                return False, "trait_injection.target_agent_ids must be integers"

    return True, ""
