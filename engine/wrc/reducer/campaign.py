"""Discover and reduce serialized WRC row/Atlas runtime material."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from .aggregate import ParameterReductionError, reduce_parameter
from .config import ReducerConfig
from .core import WorldReducer
from .model import ReducerInputError, WorldIdentity
from .output import sha256_file, write_atlas_outputs, write_row_outputs
from .providers import (
    MetricsCsvObservationProvider,
    ObservationProviderError,
    PathTextResource,
    TextResource,
    ZipTextResource,
)


class CampaignReductionError(ValueError):
    """A WRC runtime tree cannot be discovered or reduced safely."""


class _Storage:
    kind: str

    def exists(self, relative: str) -> bool:
        raise NotImplementedError

    def read_text(self, relative: str) -> str:
        raise NotImplementedError

    def resource(self, relative: str) -> TextResource:
        raise NotImplementedError

    def paths(self) -> Sequence[str]:
        raise NotImplementedError

    def source_description(self) -> dict[str, Any]:
        raise NotImplementedError


class _DirectoryStorage(_Storage):
    kind = "directory"

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _path(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise CampaignReductionError(f"unsafe relative path: {relative}") from exc
        return candidate

    def exists(self, relative: str) -> bool:
        return self._path(relative).is_file()

    def read_text(self, relative: str) -> str:
        path = self._path(relative)
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CampaignReductionError(f"cannot read {path}: {exc}") from exc

    def resource(self, relative: str) -> TextResource:
        return PathTextResource(self._path(relative))

    def paths(self) -> Sequence[str]:
        return [
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        ]

    def source_description(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": str(self.root)}


class _ZipStorage(_Storage):
    kind = "zip"

    def __init__(self, archive: Path) -> None:
        self.archive = archive.resolve()
        try:
            with zipfile.ZipFile(self.archive) as bundle:
                self._paths = tuple(
                    sorted(name for name in bundle.namelist() if not name.endswith("/"))
                )
        except (OSError, zipfile.BadZipFile) as exc:
            raise CampaignReductionError(f"cannot read ZIP {archive}: {exc}") from exc
        self._path_set = set(self._paths)
        self._sha256 = sha256_file(self.archive)

    def exists(self, relative: str) -> bool:
        return relative in self._path_set

    def read_text(self, relative: str) -> str:
        try:
            with zipfile.ZipFile(self.archive) as bundle:
                return bundle.read(relative).decode("utf-8", errors="replace")
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise CampaignReductionError(
                f"cannot read {self.archive}!/{relative}: {exc}"
            ) from exc

    def resource(self, relative: str) -> TextResource:
        if not self.exists(relative):
            raise CampaignReductionError(
                f"ZIP member is missing: {self.archive}!/{relative}"
            )
        return ZipTextResource(self.archive, relative)

    def paths(self) -> Sequence[str]:
        return self._paths

    def source_description(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": str(self.archive),
            "bytes": self.archive.stat().st_size,
            "sha256": self._sha256,
        }


@dataclass(frozen=True, slots=True)
class RowSource:
    storage: _Storage
    prefix: str

    def _join(self, relative: str) -> str:
        return str(PurePosixPath(self.prefix) / relative) if self.prefix else relative

    def exists(self, relative: str) -> bool:
        return self.storage.exists(self._join(relative))

    def read_text(self, relative: str) -> str:
        return self.storage.read_text(self._join(relative))

    def read_json(self, relative: str) -> dict[str, Any]:
        path = self._join(relative)
        try:
            value = json.loads(self.storage.read_text(path))
        except json.JSONDecodeError as exc:
            raise CampaignReductionError(f"malformed JSON {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise CampaignReductionError(f"JSON root must be an object: {path}")
        return value

    def resource(self, relative: str) -> TextResource:
        return self.storage.resource(self._join(relative))

    @property
    def source_uri(self) -> str:
        description = self.storage.source_description()
        return f"{description['kind']}://{description['path']}#/{self.prefix}"


def _storage(path: str | Path) -> _Storage:
    source = Path(path)
    if source.is_dir():
        return _DirectoryStorage(source)
    if source.is_file() and source.suffix.lower() == ".zip":
        return _ZipStorage(source)
    raise CampaignReductionError(f"input is not a directory or ZIP: {source}")


def discover_rows(
    input_path: str | Path,
    *,
    row_ids: Iterable[str] = (),
) -> list[RowSource]:
    storage = _storage(input_path)
    filters = {str(value) for value in row_ids if str(value)}
    suffix = "/_launcher/launch_plan.json"
    prefixes: set[str] = set()
    for path in storage.paths():
        normalized = path.strip("/")
        if normalized == "_launcher/launch_plan.json":
            prefixes.add("")
        elif normalized.endswith(suffix):
            prefixes.add(normalized[: -len(suffix)])
    rows: list[tuple[str, RowSource]] = []
    for prefix in sorted(prefixes):
        source = RowSource(storage, prefix)
        plan = source.read_json("_launcher/launch_plan.json")
        row_id = str(plan.get("id") or "")
        if not row_id:
            raise CampaignReductionError(
                f"row launch plan has no ID: {source.source_uri}"
            )
        if filters and row_id not in filters:
            continue
        rows.append((row_id, source))
    if filters:
        found = {row_id for row_id, _ in rows}
        missing = sorted(filters - found)
        if missing:
            raise CampaignReductionError(f"requested rows are absent: {missing}")
    return [source for _, source in sorted(rows, key=lambda item: item[0])]


def _activation_metrics(plan: Mapping[str, Any]) -> set[str]:
    challenge = plan.get("challenge")
    assertions = challenge.get("assertions", []) if isinstance(challenge, Mapping) else []
    return {
        str(row["metric"])
        for row in assertions
        if isinstance(row, Mapping) and row.get("metric")
    }


def _world_identity(
    *,
    plan: Mapping[str, Any],
    selected_row: Mapping[str, Any],
    job: Mapping[str, Any],
    source_uri: str,
) -> WorldIdentity:
    full = plan.get("full") if isinstance(plan.get("full"), Mapping) else {}
    return WorldIdentity(
        world_id=str(job["run_id"]),
        parameter_id=str(plan.get("parameter") or selected_row.get("Parameter")),
        parameter_value=job.get("level"),
        seed=int(job["seed"]),
        ticks_per_observation=int(full.get("ticks_per_generation") or 1),
        metadata={
            "row_id": str(plan.get("id") or selected_row.get("ID") or ""),
            "family": str(selected_row.get("Family") or ""),
            "expected_observations": int(full.get("generations") or 0),
            "source_row_uri": source_uri,
        },
    )


def reduce_row_source(
    source: RowSource,
    *,
    output_dir: Path,
    config: ReducerConfig | None = None,
    allow_partial_worlds: bool = False,
) -> dict[str, Any]:
    resolved_config = config or ReducerConfig()
    plan = source.read_json("_launcher/launch_plan.json")
    selected_row = source.read_json("_launcher/selected_row.json")
    full = plan.get("full")
    if not isinstance(full, Mapping) or not isinstance(full.get("jobs"), list):
        raise CampaignReductionError("launch plan lacks full jobs")
    expected_observations = int(full.get("generations") or 0)
    ticks_per_observation = int(full.get("ticks_per_generation") or 1)
    required_metrics = _activation_metrics(plan)
    world_results: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []

    for job in full["jobs"]:
        if not isinstance(job, Mapping):
            raise CampaignReductionError("launch plan job is malformed")
        run_id = str(job.get("run_id") or "")
        if not run_id:
            raise CampaignReductionError("launch plan job has no run_id")
        relative_metrics = f"full/{run_id}/metrics.csv"
        if not source.exists(relative_metrics):
            incomplete.append(
                {
                    "run_id": run_id,
                    "level": job.get("level"),
                    "seed": job.get("seed"),
                    "reason": "metrics_missing",
                    "source": source.source_uri,
                }
            )
            continue
        identity = _world_identity(
            plan=plan,
            selected_row=selected_row,
            job=job,
            source_uri=source.source_uri,
        )
        provider = MetricsCsvObservationProvider(
            source.resource(relative_metrics),
            ticks_per_observation=ticks_per_observation,
            config=resolved_config,
            required_metrics=required_metrics,
            allow_partial_last_row=True,
        )
        reducer = WorldReducer(identity, config=resolved_config)
        try:
            reducer.ingest_many(provider)
            result = reducer.finalize()
        except (ObservationProviderError, ReducerInputError) as exc:
            incomplete.append(
                {
                    "run_id": run_id,
                    "level": job.get("level"),
                    "seed": job.get("seed"),
                    "reason": "reduction_error",
                    "error": str(exc),
                    "source": source.source_uri,
                }
            )
            continue
        if expected_observations and result["observation_count"] < expected_observations:
            incomplete.append(
                {
                    "run_id": run_id,
                    "level": job.get("level"),
                    "seed": job.get("seed"),
                    "reason": "metrics_incomplete",
                    "expected_observations": expected_observations,
                    "observed_observations": result["observation_count"],
                    "source": source.source_uri,
                }
            )
            if not allow_partial_worlds:
                continue
        world_results.append(result)

    if not world_results:
        raise CampaignReductionError(
            f"row has no reducible worlds: {plan.get('id')}: {source.source_uri}"
        )
    try:
        parameter_result = reduce_parameter(
            selected_row=selected_row,
            launch_plan=plan,
            world_results=world_results,
            config=resolved_config,
        )
    except ParameterReductionError as exc:
        raise CampaignReductionError(
            f"cannot aggregate row {plan.get('id')}: {exc}"
        ) from exc

    source_description = {
        **source.storage.source_description(),
        "row_prefix": source.prefix,
        "row_uri": source.source_uri,
    }
    return write_row_outputs(
        output_dir,
        selected_row=selected_row,
        launch_plan=plan,
        world_results=world_results,
        parameter_result=parameter_result,
        incomplete_worlds=incomplete,
        config=resolved_config,
        source_description=source_description,
    )


def reduce_input(
    input_path: str | Path,
    *,
    output_dir: str | Path,
    row_ids: Iterable[str] = (),
    config: ReducerConfig | None = None,
    allow_partial_worlds: bool = False,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    resolved_config = config or ReducerConfig()
    source_path = Path(input_path)
    rows = discover_rows(source_path, row_ids=row_ids)
    if not rows:
        raise CampaignReductionError(f"no WRC rows found in {source_path}")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    row_results: list[dict[str, Any]] = []
    master_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for source in rows:
        plan = source.read_json("_launcher/launch_plan.json")
        row_id = str(plan.get("id") or "unknown")
        selected = source.read_json("_launcher/selected_row.json")
        profile = str(selected.get("Launcher profile") or plan.get("launcher_profile") or "")
        if profile in {"SPECIALIZED_MANUAL", "CONTEXT_MANUAL"}:
            skipped_rows.append(
                {
                    "id": row_id,
                    "profile": profile,
                    "reason": "row is outside generic runtime-metrics reduction",
                }
            )
            continue
        row_output = output_root / "rows" / row_id
        try:
            result = reduce_row_source(
                source,
                output_dir=row_output,
                config=resolved_config,
                allow_partial_worlds=allow_partial_worlds,
            )
        except CampaignReductionError as exc:
            failed_rows.append(
                {"id": row_id, "error": str(exc), "source": source.source_uri}
            )
            if not continue_on_error:
                raise
            continue
        row_results.append(
            {
                "id": row_id,
                # Stable, output-root-relative locator.  The absolute output
                # directory remains available in the command result, but it
                # is deliberately excluded from deterministic artifacts.
                "output_dir": str(Path("rows") / row_id),
                "manifest_sha256": result["manifest"]["manifest_sha256"],
            }
        )
        stable_master_row = dict(result["master_row"])
        row_prefix = Path("rows") / row_id
        for key in (
            "parameter_card_path",
            "parameter_card_markdown_path",
            "parameter_summary_path",
        ):
            value = stable_master_row.get(key)
            if value:
                stable_master_row[key] = str(row_prefix / Path(str(value)).name)
        master_rows.append(stable_master_row)

    storage = rows[0].storage
    atlas_manifest = write_atlas_outputs(
        output_root,
        master_rows=master_rows,
        row_results=row_results,
        skipped_rows=skipped_rows,
        failed_rows=failed_rows,
        config=resolved_config,
        source_description=storage.source_description(),
    )
    return {
        "output_dir": str(output_root),
        "row_result_count": len(row_results),
        "skipped_row_count": len(skipped_rows),
        "failed_row_count": len(failed_rows),
        "manifest": atlas_manifest,
    }


__all__ = [
    "CampaignReductionError",
    "RowSource",
    "discover_rows",
    "reduce_input",
    "reduce_row_source",
]
