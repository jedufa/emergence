#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROFILE_CORPUS = {
    "d1": "d1_fixture_corpus_v1.json",
    "d3": "d3_fixture_corpus_v1.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _without_d1_environment(payload: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(json.dumps(payload, sort_keys=True))
    data.pop("environment", None)
    scope = data.get("qualified_scope")
    if isinstance(scope, dict):
        scope.pop("verifier_torch_version", None)
    for row in data.get("fixtures", []):
        if isinstance(row, dict):
            source = row.get("source")
            if isinstance(source, dict):
                source.pop("torch_version", None)
    return data


def _without_d3_environment(payload: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(json.dumps(payload, sort_keys=True))
    data.pop("math_environment", None)
    return data


def _normalize_corpus(profile: str, payload: dict[str, Any]) -> dict[str, Any]:
    if profile == "d1":
        return _without_d1_environment(payload)
    if profile == "d3":
        return _without_d3_environment(payload)
    raise ValueError(f"unsupported profile: {profile}")


def compare_fixture_directories(
    *,
    profile: str,
    accepted: Path,
    staged: Path,
) -> dict[str, Any]:
    if profile not in PROFILE_CORPUS:
        raise ValueError(f"profile must be one of {sorted(PROFILE_CORPUS)}")
    if not accepted.is_dir():
        raise FileNotFoundError(f"accepted fixture directory is missing: {accepted}")
    if not staged.is_dir():
        raise FileNotFoundError(f"staged fixture directory is missing: {staged}")

    corpus_name = PROFILE_CORPUS[profile]
    accepted_files = {
        path.relative_to(accepted).as_posix(): path
        for path in sorted(accepted.rglob("*"))
        if path.is_file()
    }
    staged_files = {
        path.relative_to(staged).as_posix(): path
        for path in sorted(staged.rglob("*"))
        if path.is_file()
    }
    all_names = sorted(set(accepted_files) | set(staged_files))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for name in all_names:
        accepted_path = accepted_files.get(name)
        staged_path = staged_files.get(name)
        row: dict[str, Any] = {
            "path": name,
            "accepted_exists": accepted_path is not None,
            "staged_exists": staged_path is not None,
        }
        if accepted_path is None or staged_path is None:
            row["status"] = "missing"
            failures.append(row)
            rows.append(row)
            continue
        row["accepted_sha256"] = _sha256(accepted_path)
        row["staged_sha256"] = _sha256(staged_path)
        if name == "MANIFEST.sha256":
            row["status"] = "ignored_manifest_recomputed_from_corpus_metadata"
        elif name == corpus_name:
            accepted_json = json.loads(accepted_path.read_text(encoding="utf-8"))
            staged_json = json.loads(staged_path.read_text(encoding="utf-8"))
            accepted_normalized = _normalize_corpus(profile, accepted_json)
            staged_normalized = _normalize_corpus(profile, staged_json)
            row["accepted_normalized_sha256"] = _json_sha256(accepted_normalized)
            row["staged_normalized_sha256"] = _json_sha256(staged_normalized)
            if accepted_normalized == staged_normalized:
                row["status"] = "metadata_ignored_payload_match"
            else:
                row["status"] = "normalized_payload_mismatch"
                failures.append(row)
        elif row["accepted_sha256"] == row["staged_sha256"]:
            row["status"] = "byte_identical"
        else:
            row["status"] = "byte_mismatch"
            failures.append(row)
        rows.append(row)

    return {
        "schema_version": "eiel_option_d_fixture_compare_v1",
        "profile": profile,
        "accepted": str(accepted.resolve()),
        "staged": str(staged.resolve()),
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "rows": rows,
        "authority": (
            "metadata-aware production fixture comparison only; "
            "does not grant CUDA or production integration authority"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILE_CORPUS), required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = compare_fixture_directories(
        profile=args.profile,
        accepted=args.accepted,
        staged=args.staged,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "profile": receipt["profile"],
                "passed": receipt["passed"],
                "failure_count": receipt["failure_count"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
