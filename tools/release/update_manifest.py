"""Refresh the public manifest after reviewing every tracked source change."""
from pathlib import Path
import hashlib
import json
import subprocess

root = Path(__file__).resolve().parents[2]
paths = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0")
paths = sorted(filter(None, paths))
# The source archive uses the same explicit membership as the Git snapshot.
# Generated package metadata is checked separately after the build.
(root / "MANIFEST.in").write_text(
    "# Exact public-source membership; regenerate with update_manifest.py.\n"
    + "".join(f"include {name}\n" for name in paths)
    + "global-exclude *.pyc *.pyo *.so *.o\n"
)
rows = []
for name in paths:
    if name == "SOURCE_MANIFEST.json":
        continue
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"not a regular public source file: {name}")
    data = path.read_bytes()
    attribution = "eligible first-party scope; see LICENSE and THIRD_PARTY_NOTICES.md"
    if name.startswith("LICENSES/") or name == "LICENSE":
        attribution = "license text; its own terms apply"
    elif name.startswith("engine/tw_rngv2/"):
        attribution = "first-party and Random123/fdlibm-derived material; see native/LICENSING.md"
    elif name == "engine/option_d/native/glibc_2_43_fma_math.cuh":
        attribution = "GNU glibc-derived adaptation; LGPL-2.1-or-later"
    rows.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "attribution": attribution})
(root / "SOURCE_MANIFEST.json").write_text(json.dumps({"schema_version": "emergence_public_source_manifest_v1", "self_exclusion": "SOURCE_MANIFEST.json only; bind this file through Git or the distribution checksum", "files": rows}, indent=2) + "\n")
print(f"Manifest binds {len(rows)} public files")
