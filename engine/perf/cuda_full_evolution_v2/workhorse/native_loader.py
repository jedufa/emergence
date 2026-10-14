"""Fail-closed loader for immutable shape/backend workhorse V2 artifacts."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from .contract import (
    WORKHORSE_ABI_V2,
    WORKHORSE_EVENT_ABI_V3,
    WORKHORSE_PROFILE_FAMILY_V2,
    WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
    WORKHORSE_TRANSITION_ID_V2,
    canonical_sha256,
    workhorse_artifact_identity_v2,
    workhorse_backend_v2,
    workhorse_shape_v2,
)
from .layout import workhorse_layout_v2

WORKHORSE_BUILD_MANIFEST_SCHEMA_V2: Final[str] = (
    "eiel_full_evolution_cuda_workhorse_build_manifest_v2"
)
WORKHORSE_PREBUILT_ENV_V2: Final[str] = "EIEL_FULL_EVOLUTION_WORKHORSE_V2_PREBUILT"
WORKHORSE_REQUIRED_EXPORTS_V2: Final[tuple[str, ...]] = (
    "profile_family",
    "profile_id",
    "route_id",
    "abi_version",
    "event_abi_version",
    "semantic_transition_id",
    "semantic_transition_sha256",
    "backend_id",
    "backend_topology_sha256",
    "layout_sha256",
    "compile_shape",
    "kernel_topology",
    "runtime_inspect",
    "initialize",
    "initialize_exact",
    "prepare_graph",
    "graph_inspect",
    "release_graph",
    "advance",
    "advance_diagnostic_ablation",
    "canonicalize_events",
    "restore_derived",
    "force_causal_fixture",
    "force_birth_overflow_fixture",
    "force_middle_survivor_death_fixture",
    "force_transaction_error_fixture",
    "force_event_then_transaction_error_fixture",
)


class FullEvolutionWorkhorseLoadError(RuntimeError):
    """Raised when a workhorse artifact is not exactly source/device bound."""


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullEvolutionWorkhorseLoadError(
            f"cannot read workhorse V2 manifest: {exc}"
        ) from exc
    if type(value) is not dict:
        raise FullEvolutionWorkhorseLoadError("workhorse V2 manifest must be an object")
    return value


def expected_native_shape_v2(shape_id: str) -> dict[str, int | str | list[int]]:
    shape = workhorse_shape_v2(shape_id)
    layout = workhorse_layout_v2(shape)
    return {
        "shape_id": shape.shape_id,
        "width": shape.width,
        "height": shape.height,
        "sites": shape.sites,
        "initial_population": shape.initial_population,
        "capacity": shape.capacity,
        "index_width_bits": 32,
        "state_bytes_per_world": layout.state_bytes_per_world,
        "scratch_bytes_per_world": layout.scratch_bytes_per_world,
        "event_row_bytes": 48,
        "maximum_temporal_k": shape.temporal_k_candidates[-1],
        "temporal_k_candidates": list(shape.temporal_k_candidates),
    }


def _validate_source_inventory_rows_v2(
    inventory: object,
    *,
    project_root: Path,
    require_current_source_match: bool,
) -> list[dict[str, str]]:
    if type(require_current_source_match) is not bool:
        raise FullEvolutionWorkhorseLoadError(
            "workhorse V2 current-source policy must be exact bool"
        )
    if type(inventory) is not list or not inventory:
        raise FullEvolutionWorkhorseLoadError(
            "workhorse V2 source inventory is missing"
        )
    rows: list[dict[str, str]] = []
    for row in inventory:
        if (
            type(row) is not dict
            or set(row) != {"path", "sha256"}
            or type(row.get("path")) is not str
            or type(row.get("sha256")) is not str
            or len(row["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in row["sha256"]
            )
        ):
            raise FullEvolutionWorkhorseLoadError(
                "workhorse V2 source inventory row is invalid"
            )
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise FullEvolutionWorkhorseLoadError(
                "unsafe workhorse V2 source inventory path"
            )
        if require_current_source_match:
            source = project_root / relative
            if not source.is_file() or file_sha256(source) != row["sha256"]:
                raise FullEvolutionWorkhorseLoadError(
                    f"workhorse V2 source binding mismatch: {relative}"
                )
        rows.append(dict(row))
    return rows


def validate_workhorse_prebuilt_v2(
    artifact_path: str | Path,
    *,
    require_current_source_match: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    """Validate a sealed prebuilt.

    New execution keeps the default and must match the live checkout exactly.
    Historical receipt reopening may disable only that live-checkout comparison;
    the manifest, inventory seal, binary hash, build identity, and artifact seal
    are still recomputed and enforced.
    """
    artifact = Path(artifact_path)
    if artifact.is_symlink() or not artifact.is_file():
        raise FullEvolutionWorkhorseLoadError(
            "workhorse V2 artifact must be a regular non-symlink"
        )
    artifact = artifact.resolve()
    manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FullEvolutionWorkhorseLoadError(
            "workhorse V2 manifest must be a regular non-symlink"
        )
    manifest_path = manifest_path.resolve()
    if artifact.stat().st_mode & 0o222 or manifest_path.stat().st_mode & 0o222:
        raise FullEvolutionWorkhorseLoadError(
            "workhorse V2 artifact and manifest must be read-only"
        )
    manifest = _read_manifest(manifest_path)
    from .authority import workhorse_governing_contract_bindings_v2

    governing_contracts = workhorse_governing_contract_bindings_v2()
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("schema_version") != WORKHORSE_BUILD_MANIFEST_SCHEMA_V2:
        raise FullEvolutionWorkhorseLoadError("workhorse V2 manifest schema mismatch")
    if manifest.get("manifest_sha256") != canonical_sha256(body):
        raise FullEvolutionWorkhorseLoadError("workhorse V2 manifest hash mismatch")
    shape_id = manifest.get("shape_id")
    backend_id = manifest.get("backend_id")
    shape = workhorse_shape_v2(shape_id)
    backend = workhorse_backend_v2(backend_id)
    identity = workhorse_artifact_identity_v2(shape, backend.backend_id)
    if manifest.get("identity") != identity:
        raise FullEvolutionWorkhorseLoadError("workhorse V2 artifact identity mismatch")
    build_identity = manifest.get("build_identity")
    if type(build_identity) is not dict:
        raise FullEvolutionWorkhorseLoadError("workhorse V2 build identity is missing")
    build_body = {
        key: value
        for key, value in build_identity.items()
        if key != "build_identity_sha256"
    }
    if (
        build_identity.get("build_identity_sha256") != canonical_sha256(build_body)
        or build_identity.get("artifact_identity_sha256") != identity["identity_sha256"]
        or build_identity.get("shape_id") != shape.shape_id
        or build_identity.get("backend_id") != backend.backend_id
        or build_identity.get("target_sm") != manifest.get("target_sm")
        or build_identity.get("source_inventory_sha256")
        != manifest.get("source_inventory_sha256")
        or build_identity.get("governing_contracts") != governing_contracts
        or manifest.get("governing_contracts") != governing_contracts
    ):
        raise FullEvolutionWorkhorseLoadError("workhorse V2 build identity mismatch")
    if manifest.get("native_shape") != expected_native_shape_v2(shape.shape_id):
        raise FullEvolutionWorkhorseLoadError("workhorse V2 native shape mismatch")
    if manifest.get("kernel_topology") != backend.topology_payload(shape):
        raise FullEvolutionWorkhorseLoadError("workhorse V2 topology mismatch")
    output = manifest.get("output")
    if type(output) is not dict or output.get("file_name") != artifact.name:
        raise FullEvolutionWorkhorseLoadError("workhorse V2 output identity is missing")
    if output.get("bytes") != artifact.stat().st_size or output.get(
        "sha256"
    ) != file_sha256(artifact):
        raise FullEvolutionWorkhorseLoadError(
            "workhorse V2 output hash or size mismatch"
        )
    sealed = manifest.get("sealed_artifact_identity")
    if type(sealed) is not dict:
        raise FullEvolutionWorkhorseLoadError(
            "workhorse V2 sealed artifact identity is missing"
        )
    sealed_body = {
        key: value
        for key, value in sealed.items()
        if key != "sealed_artifact_identity_sha256"
    }
    if (
        sealed.get("sealed_artifact_identity_sha256") != canonical_sha256(sealed_body)
        or sealed.get("artifact_identity_sha256") != identity["identity_sha256"]
        or sealed.get("build_identity_sha256")
        != build_identity["build_identity_sha256"]
        or sealed.get("target_sm") != manifest.get("target_sm")
        or sealed.get("source_inventory_sha256")
        != manifest.get("source_inventory_sha256")
        or sealed.get("governing_contracts") != governing_contracts
        or sealed.get("output") != output
    ):
        raise FullEvolutionWorkhorseLoadError(
            "workhorse V2 sealed artifact identity mismatch"
        )
    if manifest.get("exported_functions") != list(WORKHORSE_REQUIRED_EXPORTS_V2):
        raise FullEvolutionWorkhorseLoadError("workhorse V2 export inventory mismatch")
    inventory = manifest.get("source_inventory")
    project_root = Path(__file__).resolve().parents[4]
    _validate_source_inventory_rows_v2(
        inventory,
        project_root=project_root,
        require_current_source_match=require_current_source_match,
    )
    if manifest.get("source_inventory_sha256") != canonical_sha256(inventory):
        raise FullEvolutionWorkhorseLoadError(
            "workhorse V2 source inventory hash mismatch"
        )
    return artifact, manifest_path, manifest


def load_workhorse_prebuilt_v2(
    artifact_path: str | Path | None = None,
    *,
    device: str | Any | None = None,
) -> ModuleType:
    supplied = artifact_path or os.environ.get(WORKHORSE_PREBUILT_ENV_V2)
    if not supplied:
        raise FullEvolutionWorkhorseLoadError(
            f"set {WORKHORSE_PREBUILT_ENV_V2} to an immutable workhorse V2 artifact"
        )
    artifact, _, manifest = validate_workhorse_prebuilt_v2(supplied)
    try:
        import torch

        if not torch.cuda.is_available():
            raise FullEvolutionWorkhorseLoadError("CUDA is unavailable")
        selected = torch.device("cuda" if device is None else device)
        if selected.type != "cuda":
            raise FullEvolutionWorkhorseLoadError("workhorse V2 requires CUDA")
        index = (
            torch.cuda.current_device() if selected.index is None else selected.index
        )
        capability = torch.cuda.get_device_capability(index)
        current_sm = f"sm_{capability[0]}{capability[1]}"
        properties = torch.cuda.get_device_properties(index)
    except FullEvolutionWorkhorseLoadError:
        raise
    except Exception as exc:
        raise FullEvolutionWorkhorseLoadError(
            f"cannot inspect CUDA device: {exc}"
        ) from exc
    if manifest.get("target_sm") != current_sm:
        raise FullEvolutionWorkhorseLoadError(
            f"workhorse V2 target {manifest.get('target_sm')} does not match {current_sm}"
        )
    module_name = manifest.get("module_name")
    if type(module_name) is not str or not module_name.isidentifier():
        raise FullEvolutionWorkhorseLoadError("workhorse V2 module name is invalid")
    spec = importlib.util.spec_from_file_location(module_name, artifact)
    if spec is None or spec.loader is None:
        raise FullEvolutionWorkhorseLoadError(
            "cannot construct workhorse V2 module spec"
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise FullEvolutionWorkhorseLoadError(
            f"cannot import workhorse V2 artifact: {exc}"
        ) from exc
    missing = [
        name for name in WORKHORSE_REQUIRED_EXPORTS_V2 if not hasattr(module, name)
    ]
    if missing:
        raise FullEvolutionWorkhorseLoadError(
            f"workhorse V2 artifact exports missing: {missing}"
        )
    shape = workhorse_shape_v2(manifest["shape_id"])
    backend = workhorse_backend_v2(manifest["backend_id"])
    identity = workhorse_artifact_identity_v2(shape, backend.backend_id)
    checks = (
        (module.profile_family(), WORKHORSE_PROFILE_FAMILY_V2, "profile family"),
        (module.profile_id(), shape.profile_id, "shape profile"),
        (module.route_id(), shape.route_id, "shape route"),
        (module.abi_version(), WORKHORSE_ABI_V2, "ABI"),
        (module.event_abi_version(), WORKHORSE_EVENT_ABI_V3, "event ABI"),
        (module.semantic_transition_id(), WORKHORSE_TRANSITION_ID_V2, "transition ID"),
        (
            module.semantic_transition_sha256(),
            WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
            "semantic hash",
        ),
        (module.backend_id(), backend.backend_id, "backend"),
        (
            module.backend_topology_sha256(),
            backend.topology_sha256(shape),
            "topology hash",
        ),
        (module.layout_sha256(), identity["layout_sha256"], "layout hash"),
        (
            module.compile_shape(),
            expected_native_shape_v2(shape.shape_id),
            "compile shape",
        ),
        (module.kernel_topology(), backend.topology_payload(shape), "topology"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise FullEvolutionWorkhorseLoadError(
                f"loaded workhorse V2 {label} mismatch"
            )
    runtime = module.runtime_inspect()
    if type(runtime) is not dict:
        raise FullEvolutionWorkhorseLoadError("native runtime inspection is missing")
    expected_runtime = {
        "device_index": index,
        "target_sm": current_sm,
        "multiprocessor_count": properties.multi_processor_count,
        "total_device_bytes": properties.total_memory,
        "backend_id": backend.backend_id,
        "profile_id": shape.profile_id,
        "route_id": shape.route_id,
    }
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            raise FullEvolutionWorkhorseLoadError(
                f"native runtime inspection mismatch for {key}"
            )
    return module


__all__ = [
    "FullEvolutionWorkhorseLoadError",
    "WORKHORSE_BUILD_MANIFEST_SCHEMA_V2",
    "WORKHORSE_PREBUILT_ENV_V2",
    "WORKHORSE_REQUIRED_EXPORTS_V2",
    "expected_native_shape_v2",
    "file_sha256",
    "load_workhorse_prebuilt_v2",
    "validate_workhorse_prebuilt_v2",
]
