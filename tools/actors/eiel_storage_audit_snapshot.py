#!/usr/bin/env python3
"""Read-only EIEL storage snapshotter.

It reads file metadata beneath explicitly supplied roots and writes compact
audit JSON only to ``--output-dir``.  It never mutates an audited root.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator


GIB = 1024 ** 3


def classify(path: Path) -> str:
    text = path.as_posix()
    name = path.name
    if "_source_cache_v2" in text:
        return "source_cache"
    if "source_projections_v2" in text:
        return "run_projection_copy"
    if "checkpoint_bundles_v2" in text:
        return "checkpoint_bundle"
    if "checkpoint" in name.lower() or "terminal_checkpoint" in text:
        return "checkpoint_projection"
    if "/objects/" in text or path.suffix in {".o", ".so", ".a", ".d"}:
        return "native_build"
    if path.suffix == ".zip":
        return "zip_archive"
    if "canopy-derived" in text or "lineage-playback" in text:
        return "canopy_derived"
    if path.suffix in {".jsonl", ".log"} or "trace" in text:
        return "logs_traces"
    if path.suffix == ".pt":
        return "tensor_state"
    return "other"


def walk_files(root: Path) -> Iterator[tuple[Path, os.stat_result]]:
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            yield Path(entry.path), entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
        except OSError:
            continue


def rows(root: Path, now: float) -> dict:
    totals = Counter()
    counts = Counter()
    extensions = Counter()
    extension_counts = Counter()
    ages = Counter()
    age_counts = Counter()
    top_children = Counter()
    top_child_counts = Counter()
    changed_children = Counter()
    changed_child_counts = Counter()
    directories = Counter()
    directory_counts = Counter()
    largest: list[dict] = []
    apparent = allocated = 0
    files = 0
    for path, stat in walk_files(root):
        size = int(stat.st_size)
        files += 1
        apparent += size
        allocated += int(stat.st_blocks) * 512
        category = classify(path)
        totals[category] += size
        counts[category] += 1
        extension = path.suffix.lower() or "[none]"
        extensions[extension] += size
        extension_counts[extension] += 1
        age = now - stat.st_mtime
        if age < 86400:
            age_bucket = "lt_24h"
        elif age < 3 * 86400:
            age_bucket = "1_3d"
        elif age < 7 * 86400:
            age_bucket = "3_7d"
        elif age < 30 * 86400:
            age_bucket = "7_30d"
        else:
            age_bucket = "gt_30d"
        ages[age_bucket] += size
        age_counts[age_bucket] += 1
        try:
            relative = path.relative_to(root)
            child = relative.parts[0] if relative.parts else "."
            for depth in range(1, len(relative.parts)):
                directory = "/".join(relative.parts[:depth])
                directories[directory] += size
                directory_counts[directory] += 1
        except ValueError:
            child = "."
        top_children[child] += size
        top_child_counts[child] += 1
        if age < 86400:
            changed_children[child] += size
            changed_child_counts[child] += 1
        largest.append({"path": str(path), "bytes": size, "mtime_epoch": stat.st_mtime, "class": category})
    def ranked(values: Counter, number: Counter, limit: int | None = None) -> list[dict]:
        result = [
            {"name": name, "bytes": value, "files": int(number.get(name, 0))}
            for name, value in values.items()
        ]
        result.sort(key=lambda item: (-item["bytes"], item["name"]))
        return result if limit is None else result[:limit]
    largest.sort(key=lambda item: (-item["bytes"], item["path"]))
    return {
        "root": str(root), "files": files, "apparent_bytes": apparent,
        "allocated_bytes": allocated, "allocated_gib": allocated / GIB,
        "by_artifact_class": ranked(totals, counts),
        "by_extension": ranked(extensions, extension_counts),
        "by_mtime_age": ranked(ages, age_counts),
        "top_level": ranked(top_children, top_child_counts, 30),
        "largest_directories": ranked(directories, directory_counts, 30),
        "mtime_lt_24h_top_level": ranked(changed_children, changed_child_counts, 30),
        "largest_files": largest[:50],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    now = time.time()
    payload = {
        "schema_version": "eiel_storage_audit_snapshot_v1",
        "snapshot_epoch": now,
        "method": "metadata-only traversal; mtime indicates changed-file footprint, not net allocation",
        "roots": [rows(root.resolve(), now) for root in args.roots if root.is_dir()],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "storage_inventory.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
