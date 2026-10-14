"""Live CPU SimulationRunner ownership for one RECOMB-72 campaign cell."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Final, Mapping


LIVE_CELL_SCHEMA_VERSION: Final[str] = "recomb72_live_cpu_cell_v1"
CHECKPOINT_SCHEMA_VERSION: Final[str] = "recomb72_live_checkpoint_catalog_v1"
REQUIRED_SIM_CONFIG_FIELDS: Final[tuple[str, ...]] = (
    "recomb72_audit_enable",
    "recomb72_inheritance_mode",
    "recomb72_founder_pattern",
    "recomb72_audit_seed",
    "recomb72_disable_ordinary_tick_births",
    "recomb72_mutation_off",
    "recomb72_neutral_environment_enable",
    "chronicle_enable",
)


class Recomb72LiveCellError(RuntimeError):
    """The real reference runner cannot satisfy the frozen live-cell contract."""


@dataclass(frozen=True, slots=True)
class Recomb72LiveCellSpec:
    cell_id: str
    inheritance_mode: str
    founder_pattern: str
    paired_seed: int
    generations: int = 30
    ticks_per_generation: int = 80
    checkpoint_milestones: tuple[int, ...] = (5, 10, 20, 30)
    reporting_enabled: bool = True


@dataclass(frozen=True, slots=True)
class LiveRuntimeBindings:
    env_config_class: type
    pop_config_class: type
    sim_config_class: type
    runner_class: type
    logger_factory: Callable[[], Any]
    population_class: type
    crossover_owner: Callable[..., Any]
    tick_reproduction_class: type
    live_founder_hook: Callable[..., Any]
    live_generation_hook: Callable[..., Any]
    terminal_receipt: Callable[..., Mapping[str, Any]]
    checkpoint_load: Callable[..., Mapping[str, Any]]
    checkpoint_apply: Callable[..., None]


def load_live_runtime_bindings() -> LiveRuntimeBindings:
    """Load the real CPU reference runner and checkpoint owners."""

    from env.env_config import EnvConfig
    from evolution.population_pkg.core import Population
    from evolution.population_pkg.population_config import PopConfig
    from evolution.recomb72_live_fixture import (
        apply_recomb72_live_founders_v1,
        record_recomb72_live_generation_v1,
        terminal_recomb72_live_receipt_v1,
    )
    from evolution.reproduction import crossover_genome
    from simulation.checkpoint_bundle_v2 import (
        checkpoint_bundle_v2_apply,
        checkpoint_bundle_v2_load,
    )
    from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
    from simulation.runner import SimulationRunner
    from simulation.runner_pkg.runner_config import SimConfig
    from simulation.runner_pkg.runner_ticks_reproduction import TickReproductionMixin

    return LiveRuntimeBindings(
        env_config_class=EnvConfig,
        pop_config_class=PopConfig,
        sim_config_class=SimConfig,
        runner_class=SimulationRunner,
        logger_factory=NoOpMetricsLogger,
        population_class=Population,
        crossover_owner=crossover_genome,
        tick_reproduction_class=TickReproductionMixin,
        live_founder_hook=apply_recomb72_live_founders_v1,
        live_generation_hook=record_recomb72_live_generation_v1,
        terminal_receipt=terminal_recomb72_live_receipt_v1,
        checkpoint_load=checkpoint_bundle_v2_load,
        checkpoint_apply=checkpoint_bundle_v2_apply,
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _portable_projection(value: Any) -> Any:
    """Detach loader-native arrays/scalars into canonical JSON values."""

    if isinstance(value, Mapping):
        return {
            str(key): _portable_projection(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_portable_projection(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes_hex": bytes(value).hex()}
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _portable_projection(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _portable_projection(item())
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _authoritative_bundle_projection_sha256(
    payload: Mapping[str, Any],
    *,
    bundle_path: Path,
) -> str:
    """Hash trajectory authority while excluding checkpoint-parent metadata."""

    runner_state = json.loads(json.dumps(payload.get("runner_state", {})))
    # Chronicle is a passive observation surface. Its preflight estimate and
    # checkpoint-parent provenance legitimately differ across fresh-owner
    # segments and must not redefine simulation-state parity.
    runner_state.pop("scientific_chronicle_state", None)
    weights_root = Path(str(payload.get("weights_dir", "")))
    weight_hashes: dict[str, str] = {}
    if weights_root.is_dir():
        for path in sorted(item for item in weights_root.rglob("*") if item.is_file()):
            weight_hashes[path.relative_to(weights_root).as_posix()] = _file_sha256(path)
    projection = {
        "schema_version": "recomb72_authoritative_terminal_projection_v1",
        "world": payload.get("world"),
        "population": payload.get("population"),
        "rng_snapshot_v2": payload.get("rng_snapshot_v2"),
        "control_plane": payload.get("control_plane"),
        "metrics_state": payload.get("metrics_state"),
        "runner_state": runner_state,
        "bank_state": payload.get("bank_state"),
        "resident_runtime_state_v2": payload.get("resident_runtime_state_v2"),
        "native_runtime_checkpoint_projection": payload.get(
            "native_runtime_checkpoint_projection"
        ),
        "weight_files": weight_hashes,
        "bundle_generation": int(
            dict(payload.get("metadata") or {}).get("generation_index", -1)
        ),
        "bundle_name": bundle_path.name,
    }
    return hashlib.sha256(
        _canonical_bytes(_portable_projection(projection))
    ).hexdigest()


def live_source_provenance() -> dict[str, Any]:
    """Return exact source hashes for the live mechanic and executor."""

    project_root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "engine/campaigns/recomb72_live_cell.py",
        "engine/evolution/recomb72_inheritance.py",
        "engine/evolution/recomb72_live_fixture.py",
        "engine/evolution/reproduction.py",
        "engine/evolution/population_pkg/core.py",
        "engine/simulation/runner_pkg/runner_loop.py",
        "engine/simulation/runner_pkg/runner_ticks_env_apply.py",
        "engine/simulation/runner_pkg/runner_ticks_reproduction.py",
        "engine/simulation/checkpoint_bundle_v2.py",
        "engine/simulation/chronicle/observer.py",
    )
    files: dict[str, str] = {}
    for relative in relative_paths:
        path = project_root / relative
        if not path.is_file():
            raise Recomb72LiveCellError(
                f"required live source is absent: {relative}"
            )
        files[relative] = _file_sha256(path)
    return {
        "schema_version": "recomb72_live_source_provenance_v1",
        "files": files,
        "projection_sha256": hashlib.sha256(_canonical_bytes(files)).hexdigest(),
    }


def validate_live_runtime_contract(bindings: LiveRuntimeBindings) -> None:
    """Fail closed unless every required default-off runner hook exists."""

    missing_fields = [
        field
        for field in REQUIRED_SIM_CONFIG_FIELDS
        if not hasattr(bindings.sim_config_class, field)
    ]
    required_population_helpers = (
        "apply_recomb72_founder_packages",
        "_recomb72_reproduce_and_cull",
    )
    missing_population = [
        name
        for name in required_population_helpers
        if not callable(getattr(bindings.population_class, name, None))
    ]
    if missing_fields or missing_population:
        raise Recomb72LiveCellError(
            "RECOMB72_LIVE_HOOKS_ABSENT "
            f"sim_config={missing_fields!r} population={missing_population!r}"
        )
    if not callable(bindings.crossover_owner):
        raise Recomb72LiveCellError(
            "RECOMB72_LIVE_HOOKS_ABSENT public crossover owner is absent"
        )
    if not callable(getattr(bindings.tick_reproduction_class, "_handle_births", None)):
        raise Recomb72LiveCellError(
            "RECOMB72_LIVE_HOOKS_ABSENT tick-birth guard is absent"
        )
    if not all(
        callable(hook)
        for hook in (
            bindings.live_founder_hook,
            bindings.live_generation_hook,
            bindings.terminal_receipt,
        )
    ):
        raise Recomb72LiveCellError(
            "RECOMB72_LIVE_HOOKS_ABSENT live fixture API is absent"
        )


def validate_live_cell_spec(spec: Recomb72LiveCellSpec) -> None:
    from evolution.recomb72_inheritance import FOUNDER_PATTERNS, INHERITANCE_MODES

    if not spec.cell_id:
        raise Recomb72LiveCellError("live cell_id must be non-empty")
    if spec.inheritance_mode not in INHERITANCE_MODES:
        raise Recomb72LiveCellError("live cell inheritance mode is unsupported")
    if spec.founder_pattern not in FOUNDER_PATTERNS:
        raise Recomb72LiveCellError("live cell founder pattern is unsupported")
    if spec.ticks_per_generation != 80 or not 1 <= spec.generations <= 30:
        raise Recomb72LiveCellError("live cell requires G1-G30 and actual T80")
    milestones = tuple(sorted(set(int(value) for value in spec.checkpoint_milestones)))
    if milestones != spec.checkpoint_milestones:
        raise Recomb72LiveCellError("live checkpoint milestones must be sorted and unique")
    if any(value <= 0 or value > spec.generations for value in milestones):
        raise Recomb72LiveCellError("live checkpoint milestone is outside the run")


class Recomb72LiveCellExecutor:
    """Run one cell through real SimulationRunner owners and Bundle v2."""

    def __init__(self, bindings: LiveRuntimeBindings | None = None) -> None:
        self.bindings = bindings or load_live_runtime_bindings()

    def _new_runner(
        self,
        spec: Recomb72LiveCellSpec,
        *,
        cell_root: Path,
        segment_generations: int,
        fresh_start: bool,
    ) -> Any:
        bindings = self.bindings
        requested_checkpoints = {
            *spec.checkpoint_milestones,
            int(spec.generations),
        }
        checkpoint_period = (
            5
            if int(segment_generations) >= 5
            and all(generation % 5 == 0 for generation in requested_checkpoints)
            else 1
        )
        sim_cfg = bindings.sim_config_class(
            seed=int(spec.paired_seed),
            generations=int(segment_generations),
            steps_per_generation=80,
            device="cpu",
            bank_device="cpu",
            torch_threads=1,
            behavior_log_enable=False,
            policy_deterministic_actions=True,
            policy_tick_stride=1,
            chronicle_enable=True,
            chronicle_profile="scientific_full",
            chronicle_run_id=str(spec.cell_id),
            chronicle_planned_generations=int(spec.generations),
            allow_large_output_profile=True,
            checkpoint_bundle_v2_period_gen=checkpoint_period,
            checkpoint_bundle_v2_strict_replay_mode=True,
            checkpoint_rotate=0,
            recomb72_audit_enable=True,
            recomb72_inheritance_mode=str(spec.inheritance_mode),
            recomb72_founder_pattern=str(spec.founder_pattern),
            recomb72_audit_seed=int(spec.paired_seed),
            recomb72_disable_ordinary_tick_births=True,
            recomb72_mutation_off=True,
            recomb72_neutral_actions_enable=True,
            recomb72_neutral_action_id=5,
            recomb72_neutral_environment_enable=True,
            policy_weight_mutation_enable=False,
            mutation_weather_enabled=False,
            cuda_architecture_b_production_enable=False,
            genome_c1_enable=False,
            genome_c2_enable=False,
            resource_handling_representation_enable=True,
        )
        setattr(sim_cfg, "checkpoint_bundle_v2_at_start", bool(fresh_start))
        env_cfg = bindings.env_config_class(
            width=16,
            height=16,
            patch_world_enable=False,
            food_spawn_per_step=0,
            max_food=0,
            move_cost=0.0,
            rest_cost=0.0,
            gather_cost=0.0,
            attack_cost=0.0,
            food_energy=0.0,
            gru_cost_base=0.0,
            gru_cost_per_dim=0.0,
            enable_biomes=False,
            enable_combat=False,
            attack_damage=0.0,
            enable_injury=False,
            enable_scar=False,
        )
        pop_cfg = bindings.pop_config_class(
            population_size=64,
            cull_fraction=0.5,
            parent_fraction=1.0,
            slot_factor=4,
            max_slots_cap=256,
        )
        return bindings.runner_class(
            env_cfg=env_cfg,
            pop_cfg=pop_cfg,
            sim_cfg=sim_cfg,
            logger=bindings.logger_factory(),
            run_dir=str(cell_root),
            perf=False,
        )

    def _load_catalog_entry(self, root: Path, generation: int) -> dict[str, Any]:
        bundle_path = root / "checkpoint_bundles_v2" / f"bundle_gen{generation:06d}"
        if not bundle_path.is_dir():
            raise Recomb72LiveCellError(
                f"automatic Bundle v2 is absent at G{generation}: {bundle_path}"
            )
        payload = self.bindings.checkpoint_load(str(bundle_path), strict=True)
        saved_generation, checkpoint_id = self._checkpoint_identity(payload)
        if saved_generation != generation:
            raise Recomb72LiveCellError(
                f"automatic Bundle v2 has wrong generation at G{generation}"
            )
        return {
            "generation": generation,
            "bundle_path": str(bundle_path.resolve()),
            "checkpoint_id": checkpoint_id,
            "authoritative_state_sha256": _authoritative_bundle_projection_sha256(
                payload,
                bundle_path=bundle_path,
            ),
        }

    def _terminal_live_receipt(
        self,
        runner: Any,
        *,
        start_generation: int,
        terminal_generation: int,
    ) -> dict[str, Any]:
        receipt = self.bindings.terminal_receipt(runner)
        if not isinstance(receipt, Mapping):
            raise Recomb72LiveCellError(
                "terminal runner returned no RECOMB-72 live receipt"
            )
        rows = getattr(runner, "_recomb72_generation_rows", None)
        if not isinstance(rows, list):
            rows = receipt.get("generation_rows")
        if not isinstance(rows, list):
            rows = getattr(
                getattr(runner, "pop", None),
                "_recomb72_generation_boundary_events",
                None,
            )
        if not isinstance(rows, list):
            raise Recomb72LiveCellError(
                "terminal live receipt has no generation rows"
            )
        complete_rows = [
            dict(row)
            for row in rows
            if 0 <= int(row.get("generation", -1)) <= terminal_generation
        ]
        segment_rows = [
            row
            for row in complete_rows
            if start_generation < int(row["generation"]) <= terminal_generation
        ]
        row_count = receipt.get(
            "generation_row_count",
            receipt.get("observed_generation_rows"),
        )
        actual_ticks = receipt.get(
            "actual_tick_count",
            receipt.get("observed_ticks"),
        )
        if type(row_count) is not int or row_count < len(complete_rows):
            raise Recomb72LiveCellError(
                "terminal live receipt generation row count is invalid"
            )
        expected_ticks = (terminal_generation - start_generation) * 80
        restored_ticks = start_generation * 80
        if type(actual_ticks) is not int or actual_ticks - restored_ticks != expected_ticks:
            raise Recomb72LiveCellError(
                "terminal live receipt tick count does not match actual T80"
            )
        return {
            **dict(receipt),
            "start_generation": start_generation,
            "terminal_generation": terminal_generation,
            "observed_generation_rows": (
                len(complete_rows) if start_generation == 0 else len(segment_rows)
            ),
            "complete_generation_rows": len(complete_rows),
            "observed_ticks": actual_ticks - restored_ticks,
            "generation_rows": complete_rows,
        }

    @staticmethod
    def _checkpoint_identity(payload: Mapping[str, Any]) -> tuple[int, str]:
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise Recomb72LiveCellError("live checkpoint lacks metadata")
        generation = metadata.get("generation_index")
        checkpoint_id = metadata.get("checkpoint_id")
        if type(generation) is not int or not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise Recomb72LiveCellError("live checkpoint identity is malformed")
        return generation, checkpoint_id

    def execute(
        self,
        spec: Recomb72LiveCellSpec,
        cell_root: str | Path,
        *,
        resume_bundle: str | Path | None = None,
        start_generation: int = 0,
    ) -> dict[str, Any]:
        validate_live_cell_spec(spec)
        validate_live_runtime_contract(self.bindings)
        root = Path(cell_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        if not 0 <= int(start_generation) < spec.generations:
            raise Recomb72LiveCellError("live resume generation is outside the run")

        current = int(start_generation)
        pending_payload: Mapping[str, Any] | None = None
        if resume_bundle is not None:
            pending_payload = self.bindings.checkpoint_load(
                str(Path(resume_bundle).expanduser().resolve()),
                strict=True,
            )
            generation, _checkpoint_id = self._checkpoint_identity(pending_payload)
            if generation != current:
                raise Recomb72LiveCellError(
                    "live resume bundle generation does not match start_generation"
                )
        elif current != 0:
            raise Recomb72LiveCellError("live continuation requires a checkpoint bundle")

        runner = self._new_runner(
            spec,
            cell_root=root,
            segment_generations=spec.generations - current,
            fresh_start=pending_payload is None,
        )
        runner._recomb72_reporting_enabled = bool(spec.reporting_enabled)
        if pending_payload is None:
            runner.run(start_gen=0, skip_init=False)
        else:
            self.bindings.checkpoint_apply(
                runner,
                pending_payload,
                restore_rng=True,
            )
            runner.run(start_gen=current, skip_init=True)

        receipt = self._terminal_live_receipt(
            runner,
            start_generation=current,
            terminal_generation=spec.generations,
        )
        catalog_generations = sorted(
            {
                *(
                    generation
                    for generation in spec.checkpoint_milestones
                    if current < generation <= spec.generations
                ),
                spec.generations,
                *((0,) if current == 0 else ()),
            },
        )
        checkpoints = {
            str(generation): self._load_catalog_entry(root, int(generation))
            for generation in catalog_generations
        }
        terminal = checkpoints[str(spec.generations)]
        terminal_generation = int(terminal["generation"])
        terminal_digest = str(terminal["checkpoint_id"])
        terminal_authoritative_state = str(
            terminal["authoritative_state_sha256"]
        )

        return {
            "schema_version": LIVE_CELL_SCHEMA_VERSION,
            "cell_id": spec.cell_id,
            "status": "completed",
            "terminal_generation": terminal_generation,
            "ticks_executed": receipt["observed_ticks"],
            "terminal_checkpoint_id": terminal_digest,
            "terminal_authoritative_state_sha256": terminal_authoritative_state,
            "terminal_bundle_path": checkpoints[str(spec.generations)][
                "bundle_path"
            ],
            "checkpoint_catalog": checkpoints,
            "generation_events": (
                receipt["generation_rows"] if spec.reporting_enabled else []
            ),
            "transition_rows": [
                {
                    "generation": int(row["generation"]),
                    "row_sha256": str(row["row_sha256"]),
                    "actual_ticks_per_generation": int(
                        dict(row["integrity"])["actual_ticks_per_generation"]
                    ),
                }
                for row in receipt["generation_rows"]
            ],
            "terminal_live_receipt": {
                key: value
                for key, value in receipt.items()
                if key != "generation_rows"
            },
            "reporting_enabled": bool(spec.reporting_enabled),
            "chronicle_enabled": True,
            "fresh_owner_count": 1,
            "fresh_owner_continuation_exercised": resume_bundle is not None,
            "provenance": live_source_provenance(),
        }


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "LIVE_CELL_SCHEMA_VERSION",
    "LiveRuntimeBindings",
    "Recomb72LiveCellError",
    "Recomb72LiveCellExecutor",
    "Recomb72LiveCellSpec",
    "live_source_provenance",
    "load_live_runtime_bindings",
    "validate_live_cell_spec",
    "validate_live_runtime_contract",
]
