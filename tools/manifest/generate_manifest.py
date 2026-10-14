#!/usr/bin/env python3
"""
tools.generate_manifest

Stability: STABLE_CONTRACT

Purpose
-------
Generate a deterministic Phase 2.5 job manifest (manifest_v1.yaml) from a
high-level experiment specification.

Role in architecture
--------------------
This tool is part of the orchestration layer. It expands a compact YAML spec
into explicit jobs/bundles that can be executed later by the Phase 2 runner
(e.g., ``tools.run_manifest``). It must not execute simulations.

Key contracts
-------------
- Deterministic expansion: the same input spec produces identical output.
- No side effects beyond writing the manifest file(s).
- Does not modify engine behavior or configuration defaults.

Notes
-----
If a spec field is ambiguous or unsupported, this tool should fail fast with a
clear error rather than guessing.
"""

# =====================================================================
# tools/generate_manifest.py
#
# Purpose:
#   Phase 2.5 "Job Expansion Tool"
#
#   Convert a high-level experiment spec (YAML) into a flat, deterministic
#   manifest_v1.yaml for Phase 2 runner (tools/run_manifest.py).
#
#   This tool MUST NOT execute simulations.
#
# Key properties:
#   - Expands seeds (count-based or explicit list)
#   - Expands scenarios (baseline/variant by default)
#   - Generates deterministic job_ids with short content hashes
#   - Generates explicit bundles with strict baseline/variant pairing
#   - No timestamps or randomness in output generation
#
# Output schema (Manifest v1):
#   manifest_api: 1
#   git_ref: "<ref>"
#   output_root: "<path>"
#   category: "<quick|nightly|weekly|...>"
#   jobs: [...]
#   bundles: [...]
#
# Notes:
#   - Determinism depends on: identical spec bytes -> identical manifest bytes
#     (assuming same PyYAML version and Python major version).
#   - The manifest 'jobs' and 'bundles' ordering is stable and explicit.
#
# Usage:
#   python tools/generate_manifest.py --spec <spec.yaml> --out <manifest.yaml>
#
# Spec schema (High-level Spec v1):
#   spec_api: 1
#   git_ref: "baseline-104"          # required
#   output_root: "runtime/live/wrapper_jobs" # optional (default shown)
#   category: "quick"               # required (used for bundles + top-level)
#
#   # How to run:
#   registry_entry: "validation_knob_env_cfg_injury_per_damage_quick"   # optional
#   module: "experiments.runs.validation_knob_env_cfg_injury_per_damage_quick" # required if registry_entry missing
#
#   # What we're testing (metadata only, not executed by this tool):
#   knobs_under_test:
#     - "env_cfg.injury_per_damage"
#
#   # Scenario expansion:
#   scenarios: ["baseline", "variant"]  # optional, default shown
#   scenario_args:
#     baseline: { }                     # optional: merged into job args
#     variant:  { }                     # optional: merged into job args
#
#   # Seed expansion:
#   seed_policy:
#     mode: "count"                     # "count" | "explicit"
#     count: 3                          # if mode="count"
#     start: 1                          # optional (default 1) -> seeds: start..start+count-1
#     seeds: [1,2,3]                    # if mode="explicit"
#
#   # Optional extra args shared by all jobs:
#   common_args: { }                    # merged first
#
#   # Optional resources placeholders (copied into each job.resources):
#   resources:
#     device: "cuda:0"
#
#   # Optional extra meta (copied into each job.meta):
#   meta:
#     anything: "you want"
#
# =====================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyYAML is required for tools/generate_manifest.py. "
        "Install it (pip install pyyaml) in your venv."
    ) from e


# -----------------------------
# git_ref immutability preflight
# -----------------------------

_GIT_HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_HEX_SHORT_RE = re.compile(r"^[0-9a-f]{7,39}$")


def _validate_git_ref_immutable(repo_root: Path, git_ref: str) -> None:
    """Validate that git_ref is immutable under Emergence rules.

    Allowed:
      - Full 40-hex commit SHA
      - Annotated git tag (object type 'tag') that resolves to a commit

    Disallowed:
      - Short SHAs
      - Branch names
      - Any '*dirty*' marker refs
    """
    ref = (git_ref or "").strip()
    if not ref:
        raise RuntimeError("SPEC_INVALID: missing required string field 'git_ref'")

    if "dirty" in ref.lower():
        raise RuntimeError(f"SPEC_INVALID_GIT_REF: git_ref may not contain 'dirty' (got: {ref})")

    if _GIT_HEX_SHORT_RE.match(ref) and not _GIT_HEX_40_RE.match(ref):
        raise RuntimeError(
            f"SPEC_INVALID_GIT_REF: short SHA not allowed (got: {ref}). "
            "Use full 40-character commit hash from: git rev-parse HEAD"
        )

    if _GIT_HEX_40_RE.match(ref):
        # Ensure it's a commit
        try:
            t = subprocess.check_output(["git", "cat-file", "-t", ref], cwd=str(repo_root), text=True, stderr=subprocess.STDOUT).strip()
        except Exception as e:
            raise RuntimeError(f"SPEC_INVALID_GIT_REF: cannot resolve git_ref '{ref}'") from e
        if t != "commit":
            raise RuntimeError(f"SPEC_INVALID_GIT_REF: git_ref '{ref}' is not a commit (type='{t}')")
        return

    # Must be annotated tag
    try:
        t = subprocess.check_output(["git", "cat-file", "-t", ref], cwd=str(repo_root), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        raise RuntimeError(f"SPEC_INVALID_GIT_REF: cannot resolve git_ref '{ref}'") from e

    if t != "tag":
        raise RuntimeError(
            f"SPEC_INVALID_GIT_REF: git_ref '{ref}' is type '{t}', not an annotated tag. "
            "Use a full 40-char SHA or an annotated tag."
        )

    try:
        _ = subprocess.check_output(["git", "rev-parse", f"{ref}^{{commit}}"], cwd=str(repo_root), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        raise RuntimeError(f"SPEC_INVALID_GIT_REF: annotated tag '{ref}' does not resolve to a commit") from e


def _repo_root_from_this_file() -> Path:
    """Return repository root as a Path (best-effort).

    Assumes this file lives under <repo>/tools/generate_manifest.py.
    """
    return Path(__file__).resolve().parents[1]

def _validate_registry_entry_exists(repo_root: Path, registry_entry: str) -> None:
    """Validate that the registry YAML exists for a given registry_entry name."""
    name = (registry_entry or "").strip()
    if not name:
        return
    # Allow both .yaml and .yml
    p1 = repo_root / "experiments" / "registry" / "v1" / f"{name}.yaml"
    p2 = repo_root / "experiments" / "registry" / "v1" / f"{name}.yml"
    if not p1.exists() and not p2.exists():
        raise RuntimeError(
            f"SPEC_INVALID: registry_entry '{name}' not found. "
            f"Expected: {p1} (or {p2})"
        )

# -----------------------------
# Deterministic serialization
# -----------------------------

def _canonical_json(obj: Any) -> str:
    """Return a canonical JSON string for hashing.

    - sort_keys=True for stable key ordering
    - separators to avoid whitespace differences
    - ensure_ascii=True for stable encoding
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _short_hash(obj: Any, n: int = 7) -> str:
    """Compute a deterministic short hash for an object."""
    h = hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()
    return h[:n]


def _slug(s: str) -> str:
    """Make a string safe-ish for IDs (deterministic)."""
    s = s.strip()
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    s = s.replace(" ", "_")
    s = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)
    # collapse repeats
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _seed_str(seed: int) -> str:
    """Format seed as seed000001."""
    return f"seed{seed:06d}"


# -----------------------------
# YAML IO (stable output)
# -----------------------------

class _NoAliasDumper(yaml.SafeDumper):
    """Prevent YAML anchors/aliases for determinism + readability."""
    def ignore_aliases(self, data: Any) -> bool:  # type: ignore[override]
        return True


def _yaml_load(path: str) -> Dict[str, Any]:
    """Load YAML spec."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _yaml_dump(obj: Dict[str, Any]) -> str:
    """Dump YAML with stable formatting.

    Notes:
    - sort_keys=False preserves insertion order from dict construction.
    - default_flow_style=False ensures block style.
    - width large to avoid line wrapping differences.
    """
    return yaml.dump(
        obj,
        Dumper=_NoAliasDumper,
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )


# -----------------------------
# Expansion logic
# -----------------------------

def _expand_seeds(seed_policy: Dict[str, Any]) -> List[int]:
    """Expand seeds from seed_policy."""
    mode = (seed_policy.get("mode") or "count").strip().lower()

    if mode == "explicit":
        seeds = seed_policy.get("seeds")
        if not isinstance(seeds, list) or not all(isinstance(x, int) for x in seeds):
            raise RuntimeError("SPEC_INVALID: seed_policy.mode=explicit requires seed_policy.seeds: [int, ...]")
        return list(seeds)

    if mode == "count":
        count = seed_policy.get("count")
        if not isinstance(count, int) or count <= 0:
            raise RuntimeError("SPEC_INVALID: seed_policy.mode=count requires positive int seed_policy.count")
        start = seed_policy.get("start", 1)
        if not isinstance(start, int) or start < 0:
            raise RuntimeError("SPEC_INVALID: seed_policy.start must be int >= 0")
        return list(range(start, start + count))

    raise RuntimeError(f"SPEC_INVALID: seed_policy.mode must be 'count' or 'explicit' (got {mode!r})")


def _merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge dicts (b overrides a)."""
    out = dict(a)
    out.update(b)
    return out


def _make_job_id(base: str, scenario: str, seed: int, job_hash: str) -> str:
    """Generate deterministic job_id with short hash.

    Format:
      <base>__<scenario>__seed000001__<hash7>
    """
    return f"{_slug(base)}__{_slug(scenario)}__{_seed_str(seed)}__{job_hash}"


def _build_manifest(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build a Manifest v1 dict from a High-level Spec v1 dict."""

    # Required
    spec_api = spec.get("spec_api", 1)
    if spec_api != 1:
        raise RuntimeError(f"SPEC_INVALID: unsupported spec_api={spec_api!r} (expected 1)")

    git_ref = spec.get("git_ref")
    if not isinstance(git_ref, str) or not git_ref.strip():
        raise RuntimeError("SPEC_INVALID: missing required string field 'git_ref'")

    repo_root = _repo_root_from_this_file()
    _validate_git_ref_immutable(repo_root, git_ref)

    category = spec.get("category")
    if not isinstance(category, str) or not category.strip():
        raise RuntimeError("SPEC_INVALID: missing required string field 'category'")

    output_root = spec.get("output_root", "runtime/live/wrapper_jobs")
    if not isinstance(output_root, str) or not output_root.strip():
        raise RuntimeError("SPEC_INVALID: output_root must be a non-empty string")

    registry_entry = spec.get("registry_entry")
    module = spec.get("module")

    if registry_entry is not None and (not isinstance(registry_entry, str) or not registry_entry.strip()):
        raise RuntimeError("SPEC_INVALID: registry_entry must be a non-empty string when provided")

    if module is not None and (not isinstance(module, str) or not module.strip()):
        raise RuntimeError("SPEC_INVALID: module must be a non-empty string when provided")

    if not module and not registry_entry:
        raise RuntimeError("SPEC_INVALID: provide 'module' or 'registry_entry' (or both)")

    # If registry_entry is provided, ensure it exists on disk.
    if registry_entry:
        _validate_registry_entry_exists(repo_root, str(registry_entry))

# If module omitted but registry_entry present, derive common pattern
    if not module and registry_entry:
        module = f"experiments.runs.{registry_entry}"

    # Defaults
    scenarios = spec.get("scenarios", ["baseline", "variant"])
    if not isinstance(scenarios, list) or not scenarios or not all(isinstance(x, str) and x.strip() for x in scenarios):
        raise RuntimeError("SPEC_INVALID: scenarios must be a non-empty list of strings")

    scenario_args = spec.get("scenario_args", {})
    if scenario_args is None:
        scenario_args = {}
    if not isinstance(scenario_args, dict):
        raise RuntimeError("SPEC_INVALID: scenario_args must be a dict mapping scenario -> args dict")

    common_args = spec.get("common_args", {}) or {}
    if not isinstance(common_args, dict):
        raise RuntimeError("SPEC_INVALID: common_args must be a dict")

    resources = spec.get("resources", {}) or {}
    if not isinstance(resources, dict):
        raise RuntimeError("SPEC_INVALID: resources must be a dict")

    extra_meta = spec.get("meta", {}) or {}
    if not isinstance(extra_meta, dict):
        raise RuntimeError("SPEC_INVALID: meta must be a dict")

    knobs_under_test = spec.get("knobs_under_test", []) or []
    if not isinstance(knobs_under_test, list) or not all(isinstance(k, str) and k.strip() for k in knobs_under_test):
        raise RuntimeError("SPEC_INVALID: knobs_under_test must be a list of strings")

    seed_policy = spec.get("seed_policy", {"mode": "count", "count": 1, "start": 1}) or {}
    if not isinstance(seed_policy, dict):
        raise RuntimeError("SPEC_INVALID: seed_policy must be a dict")

    seeds = _expand_seeds(seed_policy)

    base_id = registry_entry or module

    # Build jobs in stable order: scenario (given order) then seed ascending
    jobs: List[Dict[str, Any]] = []
    job_ids_by_pair: Dict[Tuple[str, int], str] = {}

    for scenario in scenarios:
        per_scn_args = scenario_args.get(scenario, {}) or {}
        if not isinstance(per_scn_args, dict):
            raise RuntimeError(f"SPEC_INVALID: scenario_args[{scenario!r}] must be a dict")

        for seed in sorted(seeds):
            args = _merge_dicts(common_args, per_scn_args)

            # If seed/scenario keys already exist, respect explicit value but ensure determinism
            if "seed" not in args:
                args["seed"] = seed
            if "scenario" not in args:
                args["scenario"] = scenario

            # Hash input uses module + args + registry_entry (if provided)
            hash_input = {
                "module": module,
                "args": args,
                "meta": {"registry_entry": registry_entry} if registry_entry else {},
            }
            job_hash = _short_hash(hash_input, n=7)
            job_id = _make_job_id(base_id, scenario, int(args["seed"]), job_hash)

            job_obj: Dict[str, Any] = {
                "job_id": job_id,
                "module": module,
                "args": args,
                "meta": _merge_dicts({"registry_entry": registry_entry} if registry_entry else {}, extra_meta),
                "resources": dict(resources),
            }

            jobs.append(job_obj)
            job_ids_by_pair[(scenario, seed)] = job_id

    # Build bundles with explicit pairs (baseline/variant only when both present)
    bundles: List[Dict[str, Any]] = []
    scenario_set = set(scenarios)

    if "baseline" in scenario_set and "variant" in scenario_set and len(scenario_set) >= 2:
        pairs = []
        for seed in sorted(seeds):
            b_id = job_ids_by_pair.get(("baseline", seed))
            v_id = job_ids_by_pair.get(("variant", seed))
            if not b_id or not v_id:
                raise RuntimeError(f"SPEC_INVALID: missing explicit baseline/variant job for seed={seed}")
            pairs.append({
                "seed": seed,
                "baseline_job_id": b_id,
                "variant_job_id": v_id,
            })

        bundles.append({
            "bundle_id": registry_entry or _slug(module),
            "category": category,
            "knobs_under_test": knobs_under_test,
            "pairs": pairs,
        })

    manifest: Dict[str, Any] = {
        "manifest_api": 1,
        "git_ref": git_ref,
        "output_root": output_root,
        "category": category,
        "jobs": jobs,
        "bundles": bundles,
    }

    return manifest


# -----------------------------
# CLI
# -----------------------------

def _parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(description="Phase 2.5 manifest generator (deterministic job expansion).")
    p.add_argument("--spec", required=True, help="Path to high-level spec YAML.")
    p.add_argument("--out", required=True, help="Path to write manifest YAML.")
    p.add_argument("--also_json", default=None, help="Optional path to also write manifest JSON.")
    p.add_argument("--no-verify", action="store_true", help="Do not run tools.verify_manifest after generation")

    # Verification defaults: STRICT + VERBOSE
    # Use these to relax output/strictness when needed.
    p.add_argument("--verify-lax", action="store_true", help="Disable strict mode (warnings do NOT become errors)")
    p.add_argument("--verify-quiet", action="store_true", help="Disable verbose verification output")

    # Backward-compatible aliases (deprecated)
    p.add_argument("--verify-strict", action="store_true", help="(DEPRECATED) Same as default strict mode")
    p.add_argument("--verify-verbose", action="store_true", help="(DEPRECATED) Same as default verbose mode")

    p.add_argument("--verify-report", default="", help="Optional path to write verify_manifest JSON report")
    return p.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = _parse_args()

    spec = _yaml_load(args.spec)
    manifest = _build_manifest(spec)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    yaml_text = _yaml_dump(manifest)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(yaml_text)

    if args.also_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.also_json)) or ".", exist_ok=True)
        with open(args.also_json, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(manifest, indent=2, sort_keys=False))
            f.write("\n")
    # Verification (default: strict + verbose)
    if not bool(getattr(args, "no_verify", False)):
        try:
            from tools.verify_manifest import main as verify_main
        except Exception as e:
            raise RuntimeError("VERIFY_UNAVAILABLE: tools.verify_manifest is missing or broken") from e

        verify_argv = ["--manifest", str(args.out)]

        # Strict by default unless explicitly relaxed
        if not bool(getattr(args, "verify_lax", False)):
            verify_argv.append("--strict")
        # Verbose by default unless explicitly quiet
        if not bool(getattr(args, "verify_quiet", False)):
            verify_argv.append("--verbose")

        # Deprecated aliases (keep behavior stable if user still passes them)
        if bool(getattr(args, "verify_strict", False)) and "--strict" not in verify_argv:
            verify_argv.append("--strict")
        if bool(getattr(args, "verify_verbose", False)) and "--verbose" not in verify_argv:
            verify_argv.append("--verbose")

        if str(getattr(args, "verify_report", "") or "").strip():
            verify_argv += ["--report", str(args.verify_report)]

        rc = int(verify_main(verify_argv) or 0)
        if rc != 0:
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
