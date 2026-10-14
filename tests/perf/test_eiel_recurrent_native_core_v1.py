from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sysconfig
from pathlib import Path

import pybind11
import pytest
import torch

from engine.agents.batched_gru import forward_batched_gru
from engine.perf.recurrent_native_contract_v1 import load_recurrent_native_contract


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "engine/perf/native/recurrent"
BINDING = Path(__file__).with_name("test_eiel_recurrent_native_core_v1_binding.cpp")
MANIFEST = ROOT / "engine/perf/recurrent_native/recurrent_state_capacity_epoch_manifest_v1.json"
BUILD = Path("/tmp/emergence-test/runtime/eiel_gru_native_migration_20260720/test_build")


@pytest.fixture(scope="session")
def native_core():
    BUILD.mkdir(parents=True, exist_ok=True)
    extension = BUILD / f"_eiel_recurrent_native_core_v1{sysconfig.get_config_var('EXT_SUFFIX')}"
    command = [
        "g++",
        "-O0",
        "-g0",
        "-std=c++17",
        "-shared",
        "-fPIC",
        f"-I{pybind11.get_include()}",
        f"-I{sysconfig.get_paths()['include']}",
        f"-I{NATIVE}",
        str(BINDING),
        str(NATIVE / "stable_gru_v1.cpp"),
        str(NATIVE / "modular_recurrent_v1.cpp"),
        "-o",
        str(extension),
    ]
    subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True)
    spec = importlib.util.spec_from_file_location("_eiel_recurrent_native_core_v1", extension)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _weights(observation_size: int, seed: int = 7102) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)

    def draw(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.035

    return {
        "encW": draw(1, 64, observation_size),
        "encb": draw(1, 64),
        "Wih": draw(1, 192, 64),
        "bih": draw(1, 192),
        "Whh": draw(1, 192, 64),
        "bhh": draw(1, 192),
        "headW": draw(1, 7, 64),
        "headb": draw(1, 7),
    }


def _pack(weights: dict[str, torch.Tensor]) -> list[float]:
    fields = ("encW", "encb", "Wih", "bih", "Whh", "bhh", "headW", "headb")
    return torch.cat([weights[name].reshape(-1) for name in fields]).tolist()


def _reference(
    weights: dict[str, torch.Tensor], observation: torch.Tensor, hidden: torch.Tensor,
    active: int, aggression: float, temperature: float, drives: torch.Tensor,
    gain: float, drive_weights: tuple[float, float, float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = (torch.arange(64) < active).to(torch.float32).unsqueeze(0)
    logits, hidden_after = forward_batched_gru(
        x=observation,
        h_prev=hidden * mask,
        **weights,
        attack_bias=torch.tensor([aggression], dtype=torch.float32),
        action_temperature=torch.tensor([temperature], dtype=torch.float32),
    )
    hunger_gather, pain_rest, fatigue_rest, pain_attack, fear_attack = drive_weights
    logits[:, 4] += gain * hunger_gather * drives[0]
    logits[:, 5] += gain * (pain_rest * drives[1] + fatigue_rest * drives[2])
    logits[:, 6] -= gain * (pain_attack * drives[1] + fear_attack * drives[3])
    return logits, hidden_after * mask


@pytest.mark.parametrize("active", [0, 8, 16, 32, 64])
def test_stable_gru_one_step_matches_authoritative_batched_path(native_core, active: int) -> None:
    observation_size = 11
    weights = _weights(observation_size, seed=7102 + active)
    generator = torch.Generator().manual_seed(801 + active)
    observation = torch.randn((1, observation_size), generator=generator) * 0.2
    hidden = torch.randn((1, 64), generator=generator) * 0.15
    drives = torch.tensor([0.3, 0.2, 0.4, 0.1], dtype=torch.float32)
    drive_weights = (0.7, -0.2, 0.4, 0.15, -0.35)
    expected_logits, expected_hidden = _reference(
        weights, observation, hidden, active, 0.13, 0.8, drives, 0.6, drive_weights
    )
    actual = native_core.stable_forward(
        observation[0].tolist(), hidden[0].tolist(), _pack(weights), active,
        0.13, 0.8, drives.tolist(), 0.6, drive_weights,
    )
    contract = load_recurrent_native_contract(MANIFEST)
    assert actual["status"] == 0
    assert actual["committed_steps"] == 1
    assert actual["hidden_elements_updated"] == 64
    torch.testing.assert_close(
        torch.tensor(actual["hidden"]), expected_hidden[0], **contract.one_step_tolerance
    )
    torch.testing.assert_close(
        torch.tensor(actual["logits"]), expected_logits[0], **contract.one_step_tolerance
    )
    assert actual["action"] == int(expected_logits.argmax(dim=1)[0])


def test_stable_gru_sequence_matches_authoritative_batched_path(native_core) -> None:
    observation_size, active = 9, 64
    weights = _weights(observation_size, seed=9901)
    generator = torch.Generator().manual_seed(9902)
    reference_hidden = torch.randn((1, 64), generator=generator) * 0.1
    native_hidden = reference_hidden[0].tolist()
    drives = torch.tensor([0.1, 0.5, 0.2, 0.3], dtype=torch.float32)
    drive_weights = (0.2, 0.1, -0.2, 0.3, -0.1)
    contract = load_recurrent_native_contract(MANIFEST)
    for _ in range(24):
        observation = torch.randn((1, observation_size), generator=generator) * 0.1
        logits, reference_hidden = _reference(
            weights, observation, reference_hidden, active, -0.03, 1.1,
            drives, 0.7, drive_weights,
        )
        actual = native_core.stable_forward(
            observation[0].tolist(), native_hidden, _pack(weights), active,
            -0.03, 1.1, drives.tolist(), 0.7, drive_weights,
        )
        assert actual["status"] == 0
        native_hidden = actual["hidden"]
        torch.testing.assert_close(
            torch.tensor(native_hidden), reference_hidden[0], **contract.sequence_tolerance
        )
        torch.testing.assert_close(
            torch.tensor(actual["logits"]), logits[0], **contract.sequence_tolerance
        )


def test_stable_gru_rejects_non_ladder_active_width_without_commit(native_core) -> None:
    weights = _weights(3)
    result = native_core.stable_forward(
        [0.0] * 3, [0.0] * 64, _pack(weights), 24,
        0.0, 1.0, [0.0] * 4, 1.0, (0.0, 0.0, 0.0, 0.0, 0.0),
    )
    assert result["status"] == 3
    assert result["hidden"] == [-991.0] * 64
    assert result["logits"] == [-992.0] * 7
    assert result["action"] == -99
    assert result["committed_steps"] == 0


def test_stable_lifecycle_isolates_cell_agent_and_generation_rows(native_core) -> None:
    result = native_core.lifecycle_scenario()
    assert result["statuses"] == [0, 0, 0, 0]
    assert result["tokens"] == [101, 202, 0, 405]
    assert result["reset_rows_zero"]
    assert result["death_stale"]
    assert result["counters"] == [1, 1, 1, 1]


def test_stable_lifecycle_rejects_unknown_operation_without_mutation(native_core) -> None:
    result = native_core.invalid_lifecycle_scenario()
    assert result == {
        "statuses": [3, 3],
        "hidden_unchanged": True,
        "tokens_unchanged": True,
        "counters_unchanged": True,
    }


def test_stable_mutation_draw_tape_is_deterministic_and_fail_closed(native_core) -> None:
    first = native_core.mutation_scenario(False)
    second = native_core.mutation_scenario(False)
    assert first == second
    assert first["status"] == 0
    assert first["child"] == pytest.approx([1.1, -2.0, 3.25, -4.2])
    assert first["draws_consumed"] == 4
    malformed = native_core.mutation_scenario(True)
    assert malformed["status"] == 14
    assert malformed["child"] == [-71.0] * 4
    assert malformed["draws_consumed"] == 0


def test_stable_checkpoint_restores_fresh_owner_and_isolates_rows(native_core) -> None:
    result = native_core.checkpoint_scenario()
    assert result["statuses"] == [0, 0]
    assert result["restored_same"] and result["isolated"]
    assert result["destination_tokens"] == [111, 222, 33, 444]
    assert result["counts"] == [1, 1]


def test_stable_cost_is_exact_and_failures_do_not_commit(native_core) -> None:
    result = native_core.stable_cost(11, 32, 1.25, 0.03125)
    assert result["status"] == 0
    assert result["energy"] == 2.25
    assert result["active_hidden"] == 32
    assert result["parameters"] == 26183
    failure = native_core.stable_failure_atomic()
    assert failure == {
        "nonfinite_status": 11,
        "scratch_status": 12,
        "outputs_unchanged": True,
        "counters_unchanged": True,
    }


def test_modular_activation_routing_neutrality_mutation_inheritance_and_epoch(native_core) -> None:
    result = native_core.modular_scenario()
    assert result["statuses"] == [0] * 9
    assert result["slots"] == [0, 1]
    assert result["routed"] == pytest.approx(0.4)
    assert result["step_output"] == pytest.approx(0.1)
    assert result["neutral_duplicate"]
    assert result["dormant_residual"][5] == pytest.approx(0.1)
    assert result["cost_components"] == [2, 8, 0, 0, 2, 4, 0, 0]
    assert result["cost_total"] == 16
    assert result["mutation_deterministic"]
    assert result["flattened_routing_selection"]
    assert result["checkpoint_status"] == 0
    assert result["development_status"] == 0
    assert result["execution_order"] == [0, 1]
    assert result["development_planes"] == pytest.approx([1.0, 1.0, 1.0, 0.75])
    assert result["restore_status"] == 0
    assert result["checkpoint_exact"]
    assert result["continuation_exact"]
    assert result["inherit_statuses"] == [0, 0]
    assert result["inherit_gains"] == pytest.approx([0.25, 0.75])
    assert result["pressure_rejected_status"] == 12
    assert result["failure_atomic"]
    assert result["expanded_status"] == 0
    assert result["expanded_profile"] == 1
    assert result["source_epoch_id"] == 0
    assert result["target_epoch_id"] == 1
    assert result["epoch_generation"] == 8
    assert result["envelope_epoch_id"] == 1
    assert result["envelope_epoch_generation"] == 8
    assert result["tail_empty"]
    assert result["migration_rng_draws"] == 0


def test_modular_malformed_state_is_rejected(native_core) -> None:
    # Invalid mask, non-finite state, and dangling connection are distinct fail-closed paths.
    assert native_core.modular_malformed()["statuses"] == [8, 7, 5]


def test_modular_native_core_two_draw_mutation_bins_are_constrained(native_core) -> None:
    result = native_core.modular_mutation_bins_scenario()
    rows = {row["case_id"]: row for row in result["rows"]}

    assert result["constrained_native_core_only"] is True
    assert result["canonical_receipt_emitted"] is False
    assert result["production_c_api_exposed"] is False
    assert list(rows) == [
        "toggle_dormancy",
        "prune_eligible_module",
        "refine_descendant_parameter",
        "refine_descendant_routing_or_expression",
    ]
    assert [rows[case]["operation"] for case in rows] == [0, 1, 2, 3]
    assert all(rows[case]["status"] == 0 for case in rows)
    assert all(rows[case]["changed"] is True for case in rows)
    assert all(rows[case]["digest_changed"] is True for case in rows)
    assert rows["toggle_dormancy"]["changed_dimension"] == 1
    assert rows["prune_eligible_module"]["module_count_before"] == 2
    assert rows["prune_eligible_module"]["module_count_after"] == 1
    assert rows["refine_descendant_parameter"]["changed_dimension"] == 8
    assert (
        rows["refine_descendant_routing_or_expression"]["changed_dimension"]
        == 1 << 18
    )
    invalid = result["invalid_draw"]
    assert invalid["status"] == 1
    assert invalid["changed"] is False
    assert invalid["digest_changed"] is False


def test_modular_public_apis_reject_malformed_counts_and_preserve_execution_order(native_core) -> None:
    result = native_core.modular_hardening_scenario()
    assert result["route_count_status"] == 1
    assert result["route_output_unchanged"]
    assert result["invalid_cost_zero"]
    assert result["duplicate_status"] == 3
    assert result["duplicate_output_unchanged"]
    assert result["inherit_status"] == 3
    assert result["child_unchanged"]
    assert result["activation_statuses"] == [0, 0]
    assert result["project_status"] == 0
    assert result["projected_homologies"] == [10, 20]
    assert result["projected_execution_order"] == [1, 0]
    assert result["inactive_nan_status"] == 1
    assert result["occupied_none_status"] == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda data: data.update(extra=True), "manifest keys"),
        (lambda data: data["stable_gru"].update(hidden_width=63), "hidden_width"),
        (lambda data: data["stable_gru"].update(memory_ladder=[0, 8, 64]), "memory_ladder"),
        (lambda data: data["stable_gru"]["gate_order"].reverse(), "gate_order"),
        (lambda data: data["stable_gru"].update(packed_width_formula="drift"), "packed_width_formula"),
        (lambda data: data["stable_gru"]["checkpoint"].update(continuation="drift"), "checkpoint contract"),
        (lambda data: data["modular_recurrent"]["envelope"].update(logical_profiles={}), "capacity_profiles"),
    ],
)
def test_manifest_loader_rejects_drift(tmp_path: Path, mutation, match: str) -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutation(data)
    candidate = tmp_path / "manifest.json"
    candidate.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_recurrent_native_contract(candidate, project_root=ROOT)


def test_manifest_loader_exposes_frozen_dimensions_and_tolerances() -> None:
    contract = load_recurrent_native_contract(MANIFEST)
    assert contract.hidden_width == 64
    assert contract.memory_ladder == (0, 8, 16, 32, 64)
    assert contract.packed_width(11) == 26183
    with pytest.raises(ValueError, match="frozen action width 7"):
        contract.packed_width(11, action_width=8)
    contract.validate_memory_dim(32)
    contract.validate_state_shape(capacity=6, hidden_width=64, hmem_rows=6, active_rows=5)
    with pytest.raises(ValueError, match="outside frozen ladder"):
        contract.validate_memory_dim(24)


def test_manifest_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"first","schema_version":"second"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate JSON key: schema_version"):
        load_recurrent_native_contract(duplicate, project_root=ROOT)


def test_native_sources_are_framework_free_cxx17_warning_clean() -> None:
    subprocess.run(
        ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
         f"-I{NATIVE}", str(NATIVE / "stable_gru_v1.cpp"),
         str(NATIVE / "modular_recurrent_v1.cpp")],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
