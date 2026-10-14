#!/usr/bin/env python3
"""Build immutable repeat sources for the frozen Genome V2 hereditary run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Final, Mapping, Sequence
from uuid import uuid4


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.authoritative_state.schema import (  # noqa: E402
    content_sha256,
    file_sha256,
)
from engine.evolution.genome_upgrade_v2_hereditary_capacity_gate_v1 import (  # noqa: E402
    validate_authority_v1,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (  # noqa: E402
    package_identity_sha256_v1,
)
from tools.scripts.genome_upgrade_v2_hereditary_capacity_collect_v1 import (  # noqa: E402
    EXECUTION_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
)


NATIVE_RUNTIME_CHECKPOINT_PROJECTION_FILENAME: Final[str] = (
    "native_runtime_checkpoint_projection.json"
)
NATIVE_RUNTIME_CHECKPOINT_PROJECTION_GZIP_FILENAME: Final[str] = (
    "native_runtime_checkpoint_projection.json.gz"
)
SCHEMA_VERSION: Final[str] = (
    "genome_v2_p10_hereditary_capacity_source_builder_v1"
)
CAMPAIGN_ID: Final[str] = "genome-v2-p10-hereditary-capacity-b2g3"
PROVIDER_ID: Final[str] = "eiel-production-provider-v1"
PROFILE_ID: Final[str] = "eiel-production-runtime-v1"
EXPECTED_ECOLOGY_SEEDS: Final[tuple[int, int]] = (101, 202)
EXPECTED_TASK_SEED: Final[int] = 3003
EXPECTED_GENERATIONS: Final[int] = 3
EXPECTED_TICKS: Final[int] = 80
EXPECTED_CHECKPOINTS: Final[tuple[int, int]] = (0, 3)
EXPECTED_PARAMETERS: Final[dict[str, Any]] = {
    "patch.grid_size": 16,
    "patch.initial_population": 64,
    "patch.capacity": 256,
    "patch.founder_placement": "balanced",
    "genome_upgrade_v2.profile": "genome_upgrade_v2_fixed_q10_v1",
    "genome_upgrade_v2.enable": True,
    "genome_upgrade_v2.mutation_rate_q10": 20,
    "genome_upgrade_v2.mutation_step_q10": 64,
    "genome_upgrade_v2.founder_profile": "pleiotropic_tradeoffs",
}
EXPECTED_INVARIANTS: Final[dict[str, Any]] = {
    "runtime.world_base": "patch",
    "runtime.world_overlays": "[]",
    "runtime.policy_family": "fixed_mlp",
    "runtime.selection_mode": "standard",
    "parameter.patch.grid_size": 16,
    "parameter.patch.initial_population": 64,
    "parameter.patch.capacity": 256,
    "parameter.genome_upgrade_v2.profile": (
        "genome_upgrade_v2_fixed_q10_v1"
    ),
    "parameter.genome_upgrade_v2.enable": True,
    "parameter.genome_upgrade_v2.mutation_rate_q10": 20,
    "parameter.genome_upgrade_v2.mutation_step_q10": 64,
}


class HereditaryCapacitySourceError(RuntimeError):
    """The campaign cannot form exact immutable hereditary repeat sources."""


def _fail(message: str) -> None:
    raise HereditaryCapacitySourceError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be an existing non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HereditaryCapacitySourceError(
            f"{label} is not readable JSON: {path}"
        ) from exc
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a JSON object")
    return dict(value)


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="ascii") as stream:
        json.dump(
            value,
            stream,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_plan_hash(plan: Mapping[str, Any]) -> str:
    unsigned = dict(plan)
    supplied = _sha256(
        unsigned.pop("plan_sha256", None), label="plan_sha256"
    )
    if content_sha256(unsigned) != supplied:
        _fail("execution plan canonical content hash differs")
    return supplied


def _artifact_row(
    plan: Mapping[str, Any], name: str
) -> dict[str, Any]:
    artifacts = plan.get("artifacts")
    row = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    if not isinstance(row, Mapping):
        _fail(f"execution plan omitted artifact {name}")
    path = row.get("path")
    digest = row.get("sha256")
    if not isinstance(path, str) or not path:
        _fail(f"artifact {name} has no path")
    _sha256(digest, label=f"artifact {name} hash")
    return dict(row)


def _validate_package_and_artifact(
    plan: Mapping[str, Any], authority: Mapping[str, str]
) -> None:
    artifact_row = _artifact_row(plan, "cuda_extension")
    artifact_path = Path(str(artifact_row["path"]))
    if (
        artifact_row["sha256"] != authority["artifact_sha256"]
        or artifact_path.is_symlink()
        or not artifact_path.is_file()
        or file_sha256(artifact_path) != authority["artifact_sha256"]
    ):
        _fail("CUDA artifact does not match hereditary authority")

    package_row = _artifact_row(plan, "genome_upgrade_v2_package_receipt")
    package_path = Path(str(package_row["path"]))
    if (
        package_row["sha256"] != authority["package_receipt_sha256"]
        or package_path.is_symlink()
        or not package_path.is_file()
        or file_sha256(package_path) != authority["package_receipt_sha256"]
    ):
        _fail("package receipt file does not match hereditary authority")
    package = _read_json(package_path, label="package receipt")
    identity = package.get("identity")
    if not isinstance(identity, Mapping):
        _fail("package receipt has no identity")
    if (
        package.get("ok") is not True
        or package.get("authority_state") != "package_candidate_complete"
        or package_identity_sha256_v1(package)
        != package.get("package_id_sha256")
        or package.get("package_id_sha256")
        != authority["package_id_sha256"]
        or identity.get("provider_id") != authority["provider_id"]
        or identity.get("runtime_profile_id")
        != authority["runtime_profile_id"]
        or identity.get("source_commit") != authority["source_commit"]
        or identity.get("cuda_artifact_sha256")
        != authority["artifact_sha256"]
    ):
        _fail("package identity does not match hereditary authority")

    qualification = plan.get("qualification_key")
    hashes = (
        qualification.get("artifact_hashes")
        if isinstance(qualification, Mapping)
        else None
    )
    if (
        not isinstance(hashes, Mapping)
        or hashes.get("cuda_extension") != authority["artifact_sha256"]
        or hashes.get("genome_upgrade_v2_package_receipt")
        != authority["package_receipt_sha256"]
    ):
        _fail("qualification key does not bind package authority")


def _validate_frozen_plan(
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    authority: Mapping[str, str],
) -> tuple[str, str, tuple[dict[str, Any], dict[str, Any]]]:
    plan_sha256 = _validate_plan_hash(plan)
    if (
        plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("resolved_profile") != PROFILE_ID
    ):
        _fail("execution plan is not the frozen hereditary campaign")
    execution = plan.get("execution")
    if not isinstance(execution, Mapping) or (
        execution.get("generations"),
        execution.get("ticks_per_generation"),
        execution.get("checkpoints"),
        execution.get("digest_every_generation"),
        execution.get("checkpoint_storage_mode"),
    ) != (
        EXPECTED_GENERATIONS,
        EXPECTED_TICKS,
        list(EXPECTED_CHECKPOINTS),
        True,
        "disk",
    ):
        _fail("execution plan differs from frozen g3t80 checkpoint schedule")
    output = plan.get("output")
    if (
        not isinstance(output, Mapping)
        or Path(str(output.get("runtime_root", ""))).expanduser().resolve()
        != run_root
    ):
        _fail("execution plan runtime root differs from campaign manifest")

    qualification = plan.get("qualification_key")
    invariants = (
        qualification.get("fixed_invariants")
        if isinstance(qualification, Mapping)
        else None
    )
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("backend") != "cuda_resident"
        or qualification.get("profile_id") != PROFILE_ID
        or qualification.get("admitted_batch_range") != [1, 2]
        or not isinstance(invariants, Mapping)
        or any(
            invariants.get(name) != value
            for name, value in EXPECTED_INVARIANTS.items()
        )
    ):
        _fail("execution plan qualification identity differs")

    raw_cells = plan.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 2:
        _fail("frozen hereditary plan must contain exactly two cells")
    cells: list[dict[str, Any]] = []
    for raw in raw_cells:
        if not isinstance(raw, Mapping):
            _fail("execution plan cell is not an object")
        cell = dict(raw)
        dimensions = cell.get("dimensions")
        parameters = cell.get("parameters")
        seeds = cell.get("seeds")
        if (
            not isinstance(cell.get("cell_id"), str)
            or not isinstance(dimensions, Mapping)
            or set(dimensions) != {"ecology_seed", "task_seed"}
            or dimensions.get("task_seed") != EXPECTED_TASK_SEED
            or not isinstance(parameters, Mapping)
            or dict(parameters) != EXPECTED_PARAMETERS
            or seeds
            != {
                "ecology": dimensions.get("ecology_seed"),
                "task": EXPECTED_TASK_SEED,
            }
        ):
            _fail("execution plan cell differs from frozen hereditary inputs")
        cells.append(cell)
    cells.sort(key=lambda row: int(row["dimensions"]["ecology_seed"]))
    if tuple(
        int(row["dimensions"]["ecology_seed"]) for row in cells
    ) != EXPECTED_ECOLOGY_SEEDS or len(
        {str(row["cell_id"]) for row in cells}
    ) != 2:
        _fail("execution plan ecology seeds or cell identities differ")

    batches = plan.get("physical_batches")
    if not isinstance(batches, list) or len(batches) != 1:
        _fail("frozen hereditary plan must contain one physical batch")
    batch = batches[0]
    if (
        not isinstance(batch, Mapping)
        or batch.get("ordinal") != 0
        or not isinstance(batch.get("batch_id"), str)
        or batch.get("cell_ids")
        != [row["cell_id"] for row in raw_cells]
        or set(batch["cell_ids"]) != {row["cell_id"] for row in cells}
    ):
        _fail("physical batch does not exactly cover the two plan cells")
    branches = plan.get("comparisons")
    branch_rows = (
        branches.get("branches") if isinstance(branches, Mapping) else None
    )
    if (
        not isinstance(branch_rows, list)
        or len(branch_rows) != 1
        or any(
            not isinstance(row, Mapping)
            or row.get("kind") != "reporting_parity"
            for row in branch_rows
        )
    ):
        _fail("fresh-repeat or continuation branches are not frozen")
    _validate_package_and_artifact(plan, authority)
    return plan_sha256, str(batch["batch_id"]), (cells[0], cells[1])


def _validate_campaign_manifest(
    campaign: Mapping[str, Any],
    *,
    campaign_path: Path,
    plan_sha256: str,
    batch_id: str,
    authority: Mapping[str, str],
) -> Path:
    expected_batch_path = (
        campaign_path.parent / "batches" / batch_id / "manifest.json"
    ).resolve()
    batches = campaign.get("batches")
    if (
        campaign.get("schema_version")
        != "eiel_campaign_execution_manifest_v1"
        or campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("plan_sha256") != plan_sha256
        or campaign.get("provider_id") != PROVIDER_ID
        or campaign.get("profile_id") != PROFILE_ID
        or campaign.get("execution_route_sha256")
        != authority["execution_route_sha256"]
        or campaign.get("status") != "completed"
        or not isinstance(batches, list)
        or len(batches) != 1
        or not isinstance(batches[0], Mapping)
        or batches[0].get("batch_id") != batch_id
        or batches[0].get("status") != "completed"
        or Path(str(batches[0].get("manifest_path", ""))).resolve()
        != expected_batch_path
    ):
        _fail("campaign manifest is not the exact completed frozen run")
    return expected_batch_path


def _checkpoint_paths(
    result: Mapping[str, Any],
    *,
    batch_root: Path,
    batch_id: str,
    plan_sha256: str,
    cell_ids: Sequence[str],
    authority: Mapping[str, str],
) -> dict[str, dict[int, Path]]:
    catalog = result.get("checkpoint_catalog")
    if not isinstance(catalog, Mapping) or set(catalog) != {"0", "3"}:
        _fail("checkpoint catalog must contain exactly G0 and G3")
    expected_cell_ids = set(cell_ids)
    selected = {cell_id: {} for cell_id in cell_ids}

    def _sidecar_path(bundle: Path) -> Path:
        compressed = bundle / NATIVE_RUNTIME_CHECKPOINT_PROJECTION_GZIP_FILENAME
        if compressed.exists():
            return compressed
        return bundle / NATIVE_RUNTIME_CHECKPOINT_PROJECTION_FILENAME

    for generation in EXPECTED_CHECKPOINTS:
        checkpoint = catalog[str(generation)]
        binding = (
            checkpoint.get("checkpoint_binding")
            if isinstance(checkpoint, Mapping)
            else None
        )
        artifact_hashes = (
            binding.get("artifact_sha256s")
            if isinstance(binding, Mapping)
            else None
        )
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("generation") != generation
            or checkpoint.get("execution_route_sha256")
            != authority["execution_route_sha256"]
            or not isinstance(binding, Mapping)
            or binding.get("provider_id") != PROVIDER_ID
            or binding.get("profile_id") != PROFILE_ID
            or binding.get("plan_sha256") != plan_sha256
            or binding.get("batch_id") != batch_id
            or binding.get("cell_ids") != list(cell_ids)
            or binding.get("execution_route_sha256")
            != authority["execution_route_sha256"]
            or not isinstance(artifact_hashes, Mapping)
            or artifact_hashes.get("cuda_extension")
            != authority["artifact_sha256"]
            or artifact_hashes.get("genome_upgrade_v2_package_receipt")
            != authority["package_receipt_sha256"]
        ):
            _fail(f"G{generation} checkpoint binding differs")
        rows = checkpoint.get("cell_bundles")
        terminal_digests = checkpoint.get("terminal_cell_digests")
        if (
            not isinstance(rows, list)
            or len(rows) != len(cell_ids)
            or not isinstance(terminal_digests, Mapping)
            or set(terminal_digests) != expected_cell_ids
        ):
            _fail(f"G{generation} checkpoint cell coverage differs")
        by_id = {
            row.get("cell_id"): row
            for row in rows
            if isinstance(row, Mapping)
            and isinstance(row.get("cell_id"), str)
        }
        if set(by_id) != expected_cell_ids:
            _fail(f"G{generation} checkpoint cell identities differ")
        for cell_id in cell_ids:
            projection_sha256 = _sha256(
                by_id[cell_id].get("source_projection_sha256"),
                label=f"G{generation} {cell_id} source projection",
            )
            if (
                _sha256(
                    terminal_digests.get(cell_id),
                    label=f"G{generation} {cell_id} terminal digest",
                )
                != projection_sha256
            ):
                _fail(f"G{generation} source projection digest differs")
            bundle = Path(str(by_id[cell_id].get("path", "")))
            expected = (
                batch_root
                / "checkpoint_bundles_v2"
                / cell_id
                / f"bundle_gen{generation:06d}"
            )
            if (
                not bundle.is_absolute()
                or bundle.is_symlink()
                or bundle.resolve() != expected.resolve()
                or not bundle.is_dir()
            ):
                _fail(f"G{generation} bundle path escapes expected batch cell")
            relative = bundle.relative_to(batch_root)
            cursor = batch_root
            if any(
                (cursor := cursor / part).is_symlink()
                for part in relative.parts
            ):
                _fail(f"G{generation} bundle path contains a symlink")
            sidecar = _sidecar_path(bundle)
            if sidecar.is_symlink() or not sidecar.is_file():
                _fail(f"G{generation} native checkpoint sidecar is absent")
            selected[cell_id][generation] = sidecar.resolve()
    return selected


def _validate_batch(
    batch: Mapping[str, Any],
    *,
    batch_path: Path,
    plan_sha256: str,
    batch_id: str,
    cells: Sequence[Mapping[str, Any]],
    authority: Mapping[str, str],
) -> tuple[str, dict[str, dict[int, Path]]]:
    cell_ids = {str(row["cell_id"]) for row in cells}
    manifest_cell_ids = batch.get("cell_ids")
    if (
        batch.get("schema_version") != "eiel_campaign_batch_manifest_v1"
        or batch.get("campaign_id") != CAMPAIGN_ID
        or batch.get("plan_sha256") != plan_sha256
        or batch.get("provider_id") != PROVIDER_ID
        or batch.get("profile_id") != PROFILE_ID
        or batch.get("execution_route_sha256")
        != authority["execution_route_sha256"]
        or batch.get("batch_id") != batch_id
        or batch.get("status") != "completed"
        or not isinstance(manifest_cell_ids, list)
        or set(manifest_cell_ids) != cell_ids
        or len(manifest_cell_ids) != 2
        or batch.get("error") is not None
    ):
        _fail("batch manifest is not the exact completed frozen batch")
    result = batch.get("result")
    if not isinstance(result, Mapping):
        _fail("completed batch has no provider result")
    receipts = result.get("native_execution_receipts")
    if (
        result.get("execution_device") != "cuda"
        or result.get("fallback_used") is not False
        or not isinstance(receipts, list)
        or len(receipts) != 1
        or not isinstance(receipts[0], Mapping)
        or receipts[0].get("execution_device") != "cuda"
        or receipts[0].get("explicit_native_no_fallback") is not True
        or receipts[0].get("hidden_fallback") is not False
        or receipts[0].get("runtime_jit") is not False
        or receipts[0].get("start_generation") != 0
        or receipts[0].get("end_generation") != EXPECTED_GENERATIONS
    ):
        _fail("batch execution did not prove exact CUDA no-fallback g3")
    paths = _checkpoint_paths(
        result,
        batch_root=batch_path.parent,
        batch_id=batch_id,
        plan_sha256=plan_sha256,
        cell_ids=manifest_cell_ids,
        authority=authority,
    )
    return file_sha256(batch_path), paths


def build_sources_v1(
    *,
    authority_path: Path,
    execution_plan_path: Path,
    campaign_manifest_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Validate one completed frozen run and atomically publish two sources."""

    try:
        authority = validate_authority_v1(
            _read_json(authority_path, label="hereditary authority")
        )
    except ValueError as exc:
        raise HereditaryCapacitySourceError(
            "hereditary authority is invalid"
        ) from exc
    plan = _read_json(execution_plan_path, label="execution plan")
    campaign = _read_json(campaign_manifest_path, label="campaign manifest")
    run_root = campaign_manifest_path.resolve().parent
    if execution_plan_path.resolve() != run_root / "execution_plan.json":
        _fail("execution plan is not the campaign run-root plan")
    plan_sha256, batch_id, cells = _validate_frozen_plan(
        plan, run_root=run_root, authority=authority
    )
    batch_path = _validate_campaign_manifest(
        campaign,
        campaign_path=campaign_manifest_path.resolve(),
        plan_sha256=plan_sha256,
        batch_id=batch_id,
        authority=authority,
    )
    batch = _read_json(batch_path, label="batch manifest")
    batch_sha256, checkpoints = _validate_batch(
        batch,
        batch_path=batch_path,
        plan_sha256=plan_sha256,
        batch_id=batch_id,
        cells=cells,
        authority=authority,
    )

    output = output_directory.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    temporary.mkdir()
    try:
        files = []
        for repeat_index, cell in enumerate(cells):
            cell_id = str(cell["cell_id"])
            execution = {
                "schema_version": EXECUTION_SCHEMA_VERSION,
                "plan_sha256": plan_sha256,
                "batch_manifest_sha256": batch_sha256,
                "batch_id": batch_id,
                "cell_id": cell_id,
                "artifact_sha256": authority["artifact_sha256"],
                "execution_route_sha256": authority[
                    "execution_route_sha256"
                ],
                "execution_device": "cuda",
                "explicit_native_no_fallback": True,
                "hidden_fallback": False,
                "fallback_used": False,
                "runtime_jit": False,
            }
            execution["receipt_sha256"] = content_sha256(execution)
            source = {
                "schema_version": SOURCE_SCHEMA_VERSION,
                "repeat_index": repeat_index,
                "authority": authority,
                "execution_receipt": execution,
                "observations": [
                    {
                        "generation": generation,
                        "checkpoint_path": str(
                            checkpoints[cell_id][generation]
                        ),
                        "checkpoint_sha256": file_sha256(
                            checkpoints[cell_id][generation]
                        ),
                    }
                    for generation in EXPECTED_CHECKPOINTS
                ],
            }
            filename = f"repeat-{repeat_index}.json"
            path = temporary / filename
            _write_json(path, source)
            files.append(
                {
                    "repeat_index": repeat_index,
                    "ecology_seed": cell["dimensions"]["ecology_seed"],
                    "cell_id": cell_id,
                    "path": filename,
                    "sha256": file_sha256(path),
                }
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": CAMPAIGN_ID,
            "authority": authority,
            "plan_sha256": plan_sha256,
            "campaign_manifest_sha256": file_sha256(
                campaign_manifest_path
            ),
            "batch_manifest_sha256": batch_sha256,
            "batch_id": batch_id,
            "sources": files,
        }
        manifest["manifest_sha256"] = content_sha256(manifest)
        _write_json(temporary / "source_manifest.json", manifest)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = build_sources_v1(
            authority_path=args.authority,
            execution_plan_path=args.execution_plan,
            campaign_manifest_path=args.campaign_manifest,
            output_directory=args.output_directory,
        )
    except (HereditaryCapacitySourceError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
