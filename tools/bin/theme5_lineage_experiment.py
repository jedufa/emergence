#!/usr/bin/env python3
from __future__ import annotations

"""Theme 5 lineage experiment manifest tooling.

This tool is manifest-only. It does not launch simulations, edit checkpoints,
filter populations, reseed archives, or alter live evolution.
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


SCHEMA_VERSION = "theme5_lineage_experiment_manifest_v1"
CREATED_BY = "theme5_lineage_experiment_manifest"
VALID_MANIFEST_TYPES = {
    "single_lineage_branch",
    "lineage_mix_branch",
    "archive_candidate_branch",
    "archive_mix_branch",
    "transfer_test",
    "stress_test",
}


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def stable_manifest_id(payload: Mapping[str, Any]) -> str:
    return "t5manifest_" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()[:20]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _latest_generation(records: Iterable[Mapping[str, Any]]) -> int:
    gens = [_int(r.get("generation", -1), -1) for r in records if isinstance(r, Mapping)]
    return max(gens) if gens else -1


def load_archive_entries(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    index = run_dir / "archives" / "lineage_qd" / "archive_index.jsonl"
    for row in _read_jsonl(index):
        entry_id = str(row.get("archive_entry_id", "") or "")
        if entry_id:
            entries[entry_id] = row
    entries_dir = run_dir / "archives" / "lineage_qd" / "entries"
    for entry_id in list(entries.keys()):
        path = entries_dir / f"{entry_id}.json"
        if path.exists():
            full = _read_json(path)
            if full:
                entries[entry_id] = full
    return entries


def load_candidates(run_dir: Path, *, latest_only: bool = True) -> List[Dict[str, Any]]:
    rows = _read_jsonl(run_dir / "metrics" / "branch_candidates.jsonl")
    if latest_only:
        latest = _latest_generation(rows)
        rows = [r for r in rows if _int(r.get("generation", -1), -1) == latest]
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        for candidate in row.get("candidates", []) or []:
            if isinstance(candidate, dict):
                candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda c: (-_float(c.get("candidate_score", 0.0), 0.0), str(c.get("candidate_type", "")), str(c.get("lineage_id", ""))),
    )


def known_lineage_ids(run_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in (
        run_dir / "metrics" / "lineage_metrics.jsonl",
        run_dir / "metrics" / "lineage_descriptors.jsonl",
        run_dir / "metrics" / "branch_candidates.jsonl",
    ):
        for row in _read_jsonl(path):
            if "lineage_id" in row:
                ids.add(str(row.get("lineage_id")))
            for desc in row.get("lineages", []) or []:
                if isinstance(desc, Mapping) and desc.get("lineage_id"):
                    ids.add(str(desc.get("lineage_id")))
            for candidate in row.get("candidates", []) or []:
                if isinstance(candidate, Mapping) and candidate.get("lineage_id"):
                    ids.add(str(candidate.get("lineage_id")))
    return ids


def _checkpoint_reference(run_dir: Path) -> str | None:
    candidates = _read_jsonl(run_dir / "metrics" / "branch_candidates.jsonl")
    gen = _latest_generation(candidates)
    if gen < 0:
        return None
    rel = Path("checkpoint_bundles_v2") / f"bundle_gen{gen:06d}"
    return str(rel).replace("\\", "/") if (run_dir / rel).exists() else None


def create_manifest(
    *,
    run_dir: Path,
    manifest_type: str,
    lineage_ids: Iterable[str] = (),
    archive_entry_ids: Iterable[str] = (),
    candidate_type: str = "",
    top_n: int = 0,
    min_score: float = 0.0,
    latest_only: bool = True,
    target_config_overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if manifest_type not in VALID_MANIFEST_TYPES:
        raise ValueError(f"unsupported manifest_type: {manifest_type}")

    archive_entries = load_archive_entries(run_dir)
    selected_archive_ids = sorted({str(x) for x in archive_entry_ids if str(x)})
    selected_lineages = sorted({str(x) for x in lineage_ids if str(x)})
    selection_reasons: List[str] = []
    candidate_scores: Dict[str, float] = {}
    descriptor_summary: Dict[str, Any] = {}

    for entry_id in selected_archive_ids:
        entry = archive_entries.get(entry_id, {})
        lid = str(entry.get("lineage_id", "") or "")
        if lid:
            selected_lineages.append(lid)
            selection_reasons.append(f"selected archive entry {entry_id}")
            candidate_scores[lid] = _float(entry.get("candidate_score", 0.0), 0.0)
            descriptor_summary[lid] = entry.get("descriptor_summary", {})

    if candidate_type or top_n > 0 or min_score > 0.0:
        candidates = [
            c
            for c in load_candidates(run_dir, latest_only=latest_only)
            if (not candidate_type or str(c.get("candidate_type", "")) == str(candidate_type))
            and _float(c.get("candidate_score", 0.0), 0.0) >= float(min_score)
        ]
        if top_n > 0:
            candidates = candidates[: int(top_n)]
        for candidate in candidates:
            lid = str(candidate.get("lineage_id", "") or "")
            if lid:
                selected_lineages.append(lid)
                selection_reasons.append(
                    f"selected candidate {candidate.get('candidate_type', '')} score {candidate.get('candidate_score', 0.0)}"
                )
                candidate_scores[lid] = _float(candidate.get("candidate_score", 0.0), 0.0)
                descriptor_summary[lid] = candidate.get("behavioral_descriptor_summary", {})

    selected_lineages = sorted(set(selected_lineages))
    selected_archive_ids = sorted(set(selected_archive_ids))
    target_overrides = dict(target_config_overrides or {})
    population_policy = {
        "mode": "manifest_only",
        "execution_implemented": False,
        "notes": "No population filtering, mixing, reseeding, mutation, or branch execution is performed by this manifest.",
    }
    base = {
        "manifest_type": manifest_type,
        "source_run_dir": str(run_dir),
        "selected_lineage_ids": selected_lineages,
        "selected_archive_entry_ids": selected_archive_ids,
        "candidate_type": candidate_type,
        "top_n": int(top_n),
        "min_score": float(min_score),
        "target_config_overrides": target_overrides,
    }
    manifest_id = stable_manifest_id(base)
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "manifest_type": manifest_type,
        "source_run_dir": str(run_dir),
        "source_checkpoint_reference": _checkpoint_reference(run_dir),
        "selected_lineage_ids": selected_lineages,
        "selected_archive_entry_ids": selected_archive_ids,
        "selection_reasons": sorted(set(selection_reasons)),
        "candidate_scores": {str(k): float(v) for k, v in sorted(candidate_scores.items())},
        "descriptor_summary": {str(k): descriptor_summary[k] for k in sorted(descriptor_summary.keys())},
        "intended_experiment_mode": manifest_type,
        "target_config_overrides": target_overrides,
        "population_policy": population_policy,
        "determinism_notes": [
            "manifest_id is derived from canonical deterministic selection payload",
            "manifest generation does not alter live evolution",
            "actual branch execution is deferred",
        ],
        "created_by": CREATED_BY,
        "explicit_warning": "manifest only; does not alter normal live evolution",
    }


def validate_manifest(manifest: Mapping[str, Any], *, run_dir: Path | None = None) -> Dict[str, Any]:
    issues: List[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        issues.append("unsupported schema_version")
    if str(manifest.get("manifest_type", "")) not in VALID_MANIFEST_TYPES:
        issues.append("unsupported manifest_type")
    source = Path(str(run_dir or manifest.get("source_run_dir", "") or ""))
    if not source.exists():
        issues.append("source_run_dir does not exist")
    known = known_lineage_ids(source) if source.exists() else set()
    for lid in manifest.get("selected_lineage_ids", []) or []:
        if known and str(lid) not in known:
            issues.append(f"unknown lineage_id: {lid}")
    archive = load_archive_entries(source) if source.exists() else {}
    for entry_id in manifest.get("selected_archive_entry_ids", []) or []:
        if str(entry_id) not in archive:
            issues.append(f"unknown archive_entry_id: {entry_id}")
    if not manifest.get("selected_lineage_ids") and not manifest.get("selected_archive_entry_ids"):
        issues.append("no selected lineage or archive entries")
    return {
        "schema_version": "theme5_lineage_experiment_manifest_validation_v1",
        "manifest_id": str(manifest.get("manifest_id", "")),
        "valid": not issues,
        "issues": sorted(issues),
    }


def summarize_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "theme5_lineage_experiment_manifest_summary_v1",
        "manifest_id": str(manifest.get("manifest_id", "")),
        "manifest_type": str(manifest.get("manifest_type", "")),
        "selected_lineage_count": len(manifest.get("selected_lineage_ids", []) or []),
        "selected_archive_entry_count": len(manifest.get("selected_archive_entry_ids", []) or []),
        "execution_implemented": False,
        "explicit_warning": str(manifest.get("explicit_warning", "")),
    }


def _parse_overrides(raw: str) -> Dict[str, Any]:
    if not str(raw or "").strip():
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("--target-config-overrides-json must decode to an object")
    return obj


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(manifest), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create-manifest")
    create.add_argument("--run-dir", required=True)
    create.add_argument("--manifest-type", required=True, choices=sorted(VALID_MANIFEST_TYPES))
    create.add_argument("--lineage-id", action="append", default=[])
    create.add_argument("--archive-entry-id", action="append", default=[])
    create.add_argument("--candidate-type", default="")
    create.add_argument("--top-n", type=int, default=0)
    create.add_argument("--min-score", type=float, default=0.0)
    create.add_argument("--all-generations", action="store_true")
    create.add_argument("--target-config-overrides-json", default="")
    create.add_argument("--out", default="")

    validate = sub.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--run-dir", default="")

    summarize = sub.add_parser("summarize-manifest")
    summarize.add_argument("--manifest", required=True)

    args = parser.parse_args()
    if args.command == "create-manifest":
        run_dir = Path(args.run_dir)
        manifest = create_manifest(
            run_dir=run_dir,
            manifest_type=str(args.manifest_type),
            lineage_ids=args.lineage_id,
            archive_entry_ids=args.archive_entry_id,
            candidate_type=str(args.candidate_type or ""),
            top_n=int(args.top_n or 0),
            min_score=float(args.min_score or 0.0),
            latest_only=not bool(args.all_generations),
            target_config_overrides=_parse_overrides(args.target_config_overrides_json),
        )
        out = Path(args.out) if str(args.out or "").strip() else (
            run_dir / "archives" / "lineage_qd" / "branch_manifests" / f"{manifest['manifest_id']}.json"
        )
        _write_manifest(out, manifest)
        print(str(out))
        return 0
    if args.command == "validate-manifest":
        manifest = _read_json(Path(args.manifest))
        run_dir = Path(args.run_dir) if str(args.run_dir or "").strip() else None
        print(canonical_json(validate_manifest(manifest, run_dir=run_dir)), end="")
        return 0
    if args.command == "summarize-manifest":
        print(canonical_json(summarize_manifest(_read_json(Path(args.manifest)))), end="")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
