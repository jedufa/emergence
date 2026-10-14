from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.campaigns.recomb72_live_cell import (
    LiveRuntimeBindings,
    Recomb72LiveCellError,
    Recomb72LiveCellExecutor,
    Recomb72LiveCellSpec,
    validate_live_runtime_contract,
)


class FakeSimConfig:
    recomb72_audit_enable = False
    recomb72_inheritance_mode = "current_authoritative"
    recomb72_founder_pattern = "single_extreme_scalar"
    recomb72_audit_seed = 0
    recomb72_disable_ordinary_tick_births = True
    recomb72_mutation_off = True
    recomb72_neutral_environment_enable = False
    chronicle_enable = False

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeEnvConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakePopConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakePopulation:
    def apply_recomb72_founder_packages(self):
        return {}

    def _recomb72_reproduce_and_cull(self):
        return None


class FakeTickReproduction:
    def _handle_births(self):
        return None


class FakeRunner:
    owners: list["FakeRunner"] = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.pop = SimpleNamespace(_recomb72_generation_boundary_events=[])
        self._recomb72_generation_rows = [_generation_row(0)]
        self._recomb72_real_tick_count = 0
        self.restored_generation = 0
        self.terminal_generation = 0
        self.run_calls = []
        self.__class__.owners.append(self)

    def run(self, *, start_gen: int, skip_init: bool):
        self.run_calls.append((start_gen, skip_init))
        assert self.sim_cfg.steps_per_generation == 80
        assert self.sim_cfg.chronicle_enable is True
        assert self.sim_cfg.chronicle_profile == "scientific_full"
        assert self.sim_cfg.checkpoint_bundle_v2_period_gen == 5
        assert self.sim_cfg.device == "cpu"
        assert self.env_cfg.width == self.env_cfg.height == 16
        assert self.env_cfg.patch_world_enable is False
        assert self.env_cfg.max_food == self.env_cfg.food_spawn_per_step == 0
        assert self.env_cfg.move_cost == self.env_cfg.attack_cost == 0.0
        assert self.env_cfg.gru_cost_base == self.env_cfg.gru_cost_per_dim == 0.0
        assert self.env_cfg.enable_biomes is False
        assert self.env_cfg.enable_combat is False
        assert self.env_cfg.enable_injury is False
        assert self.env_cfg.enable_scar is False
        assert self.pop_cfg.population_size == 64
        end = start_gen + self.sim_cfg.generations
        self.pop._recomb72_generation_boundary_events = [
            _generation_row(generation)
            for generation in range(start_gen + 1, end + 1)
        ]
        self._recomb72_generation_rows.extend(
            self.pop._recomb72_generation_boundary_events
        )
        self._recomb72_real_tick_count = end * 80
        self.terminal_generation = end
        checkpoint_root = Path(self.run_dir) / "checkpoint_bundles_v2"
        generations = list(range(5, end + 1, 5))
        if not skip_init and self.sim_cfg.checkpoint_bundle_v2_at_start:
            generations.insert(0, 0)
        for generation in generations:
            path = checkpoint_root / f"bundle_gen{generation:06d}"
            path.mkdir(parents=True, exist_ok=True)
            material = {
                "cell_id": self.sim_cfg.chronicle_run_id,
                "mode": self.sim_cfg.recomb72_inheritance_mode,
                "seed": self.sim_cfg.recomb72_audit_seed,
                "generation": generation,
                "owner": self.run_dir,
            }
            checkpoint_id = hashlib.sha256(
                json.dumps(material, sort_keys=True).encode()
            ).hexdigest()
            payload = {
                "metadata": {
                    "generation_index": generation,
                    "checkpoint_id": checkpoint_id,
                },
                "world": {"generation": generation, "width": 16, "height": 16},
                "population": {
                    "generation": generation,
                    "alive": 64,
                    "mode": self.sim_cfg.recomb72_inheritance_mode,
                },
                "runner_state": {
                    "generation_rows": self._recomb72_generation_rows[
                        : generation + 1
                    ],
                    "scientific_chronicle_state": {
                        "parent_checkpoint_id": checkpoint_id,
                    },
                },
            }
            (path / "payload.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )


def _generation_row(generation: int) -> dict:
    children = 0 if generation == 0 else 32
    return {
        "schema_version": "recomb72_generation_metrics_v1",
        "authority": "production_python_reference_simulation_runner",
        "generation": generation,
        "population": {
            "before_cull": 64,
            "eligible_parents": 64 if generation == 0 else 32,
            "selected_parent_events": children * 2,
            "pairings": children,
            "children": children,
            "after_refill": 64,
            "failed_refill_events": 0,
            "ordinary_births": 0,
            "mutation_events": 0,
        },
        "integrity": {
            "actual_ticks_per_generation": 80,
            "real_runner": True,
            "chronicle_enabled": True,
        },
        "row_sha256": f"{generation:064x}",
    }


def _bindings(tmp_path: Path) -> LiveRuntimeBindings:
    def checkpoint_load(bundle_dir, *, strict):
        assert strict is True
        return json.loads(
            (Path(bundle_dir) / "payload.json").read_text(encoding="utf-8")
        )

    def checkpoint_apply(runner, payload, *, restore_rng):
        assert restore_rng is True
        runner.restored_generation = payload["metadata"]["generation_index"]
        generation = runner.restored_generation
        runner._recomb72_generation_rows = [
            _generation_row(value) for value in range(generation + 1)
        ]
        runner._recomb72_real_tick_count = generation * 80

    def terminal_receipt(runner):
        return {
            "schema_version": "recomb72_live_fixture_terminal_v1",
            "actual_tick_count": runner._recomb72_real_tick_count,
            "generation_row_count": len(runner._recomb72_generation_rows),
            "generation_rows_sha256": "a" * 64,
            "chronicle_root": str(Path(runner.run_dir) / "chronicle"),
            "neutrality_receipt": {
                "trait_specific_action_authority": False,
                "ecological_selection": False,
                "genotype_sensitive_world_mechanics": False,
            },
        }

    return LiveRuntimeBindings(
        env_config_class=FakeEnvConfig,
        pop_config_class=FakePopConfig,
        sim_config_class=FakeSimConfig,
        runner_class=FakeRunner,
        logger_factory=lambda: object(),
        population_class=FakePopulation,
        crossover_owner=type(
            "FakeCrossoverOwner",
            (),
            {"crossover_genome": lambda self: None},
        ),
        tick_reproduction_class=FakeTickReproduction,
        live_founder_hook=lambda runner: None,
        live_generation_hook=lambda runner, generation: None,
        terminal_receipt=terminal_receipt,
        checkpoint_load=checkpoint_load,
        checkpoint_apply=checkpoint_apply,
    )


def test_live_executor_runs_one_uninterrupted_owner_with_automatic_bundles(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        "engine.campaigns.recomb72_live_cell.live_source_provenance",
        lambda: {"files": {}, "projection_sha256": "a" * 64},
    )
    FakeRunner.owners.clear()
    executor = Recomb72LiveCellExecutor(_bindings(tmp_path))
    spec = Recomb72LiveCellSpec(
        cell_id="cell-a",
        inheritance_mode="parent_copy_per_locus",
        founder_pattern="resource_specialization_extremes",
        paired_seed=202,
        generations=30,
        checkpoint_milestones=(5, 10, 20, 30),
    )

    result = executor.execute(spec, tmp_path / "cell")

    assert result["status"] == "completed"
    assert result["ticks_executed"] == 2400
    assert sorted(result["checkpoint_catalog"], key=int) == ["0", "5", "10", "20", "30"]
    assert result["fresh_owner_count"] == 1
    assert result["fresh_owner_continuation_exercised"] is False
    assert result["chronicle_enabled"] is True
    assert [event["generation"] for event in result["generation_events"]] == list(
        range(31)
    )
    assert result["terminal_live_receipt"]["actual_tick_count"] == 2400
    assert result["terminal_live_receipt"]["observed_generation_rows"] == 31
    assert result["terminal_live_receipt"]["generation_row_count"] == 31
    assert (
        result["terminal_live_receipt"]["neutrality_receipt"][
            "ecological_selection"
        ]
        is False
    )
    assert len(result["terminal_authoritative_state_sha256"]) == 64
    assert (
        result["terminal_authoritative_state_sha256"]
        == result["checkpoint_catalog"]["30"]["authoritative_state_sha256"]
    )
    assert (
        result["terminal_authoritative_state_sha256"]
        != result["terminal_checkpoint_id"]
    )
    assert len(FakeRunner.owners) == 1
    assert FakeRunner.owners[0].run_calls == [(0, False)]


def test_live_executor_resume_uses_fresh_owner_and_reporting_is_observational(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        "engine.campaigns.recomb72_live_cell.live_source_provenance",
        lambda: {"files": {}, "projection_sha256": "a" * 64},
    )
    bindings = _bindings(tmp_path)
    first_spec = Recomb72LiveCellSpec(
        cell_id="cell-b",
        inheritance_mode="current_authoritative",
        founder_pattern="single_extreme_scalar",
        paired_seed=101,
        generations=10,
        checkpoint_milestones=(10,),
    )
    first = Recomb72LiveCellExecutor(bindings).execute(
        first_spec, tmp_path / "first"
    )
    resumed_spec = Recomb72LiveCellSpec(
        cell_id="cell-b",
        inheritance_mode="current_authoritative",
        founder_pattern="single_extreme_scalar",
        paired_seed=101,
        generations=20,
        checkpoint_milestones=(20,),
        reporting_enabled=False,
    )
    resumed = Recomb72LiveCellExecutor(bindings).execute(
        resumed_spec,
        tmp_path / "resumed",
        resume_bundle=first["terminal_bundle_path"],
        start_generation=10,
    )

    assert resumed["terminal_generation"] == 20
    assert resumed["ticks_executed"] == 800
    assert resumed["generation_events"] == []
    assert resumed["chronicle_enabled"] is True
    assert resumed["fresh_owner_count"] == 1
    assert resumed["fresh_owner_continuation_exercised"] is True
    assert (
        resumed["terminal_authoritative_state_sha256"]
        == resumed["checkpoint_catalog"]["20"]["authoritative_state_sha256"]
    )
    assert FakeRunner.owners[-1].run_calls == [(10, True)]


def test_runtime_contract_fails_closed_when_audit_hooks_are_absent(
    tmp_path: Path,
) -> None:
    bindings = _bindings(tmp_path)

    class MissingConfig:
        chronicle_enable = False

    broken = replace(bindings, sim_config_class=MissingConfig)
    with pytest.raises(Recomb72LiveCellError, match="LIVE_HOOKS_ABSENT"):
        validate_live_runtime_contract(broken)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ticks_per_generation": 79},
        {"generations": 31},
        {"checkpoint_milestones": (10, 5)},
    ],
)
def test_live_executor_rejects_out_of_envelope_specs(
    tmp_path: Path,
    overrides,
) -> None:
    values = {
        "cell_id": "bad",
        "inheritance_mode": "current_authoritative",
        "founder_pattern": "single_extreme_scalar",
        "paired_seed": 1,
        "generations": 30,
        "ticks_per_generation": 80,
        "checkpoint_milestones": (5, 10, 20, 30),
    }
    values.update(overrides)
    with pytest.raises(Recomb72LiveCellError):
        Recomb72LiveCellExecutor(_bindings(tmp_path)).execute(
            Recomb72LiveCellSpec(**values),
            tmp_path / "bad",
        )
