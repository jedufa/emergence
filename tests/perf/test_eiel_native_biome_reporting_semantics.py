from __future__ import annotations

import pytest

from native_runtime.public_metrics_overlay import public_metrics_overlay


BIOME_FRACTION_KEYS = ("biome_safe_frac_alive", "biome_wild_frac_alive")


def _biome_fractions(tensors: dict[str, object]) -> dict[str, float]:
    overlay = public_metrics_overlay(
        checkpoint_projection={"tensors": tensors},
    )
    return {key: overlay[key] for key in BIOME_FRACTION_KEYS}


def test_final_tick_birth_remains_safe_on_wild_spawn_cell() -> None:
    fractions = _biome_fractions(
        {
            "alive_mask": [True],
            "slot_to_agent_id": [101],
            "survival_steps": [0],
            "pos_y": [0],
            "pos_x": [1],
            "biome_grid": [[0, 1]],
        }
    )

    assert fractions == {
        "biome_safe_frac_alive": 1.0,
        "biome_wild_frac_alive": 0.0,
    }


def test_stepped_live_rows_use_current_biome_grid_at_position() -> None:
    fractions = _biome_fractions(
        {
            "alive_mask": [True, True],
            "slot_to_agent_id": [101, 102],
            "survival_steps": [1, 8],
            "pos_y": [1, 0],
            "pos_x": [0, 1],
            "biome_grid": [[0, 1], [1, 0]],
            # The overlay reconstructs the reference's post-step value from
            # current positions; this stale decoy must not become authority.
            "biome_id": [0, 0],
        }
    )

    assert fractions == {
        "biome_safe_frac_alive": 0.0,
        "biome_wild_frac_alive": 1.0,
    }


def test_mixed_newborn_and_stepped_live_rows_produce_exact_fractions() -> None:
    fractions = _biome_fractions(
        {
            "alive_mask": [True, True, True, False],
            "slot_to_agent_id": [101, 102, 103, 104],
            "survival_steps": [0, 4, 9, 20],
            "pos_y": [0, 0, 1, 1],
            "pos_x": [1, 0, 0, 1],
            "biome_grid": [[0, 1], [1, 0]],
        }
    )

    assert fractions == {
        "biome_safe_frac_alive": 2.0 / 3.0,
        "biome_wild_frac_alive": 1.0 / 3.0,
    }


@pytest.mark.parametrize(
    "missing_input",
    ["survival_steps", "pos_y", "pos_x", "biome_grid"],
)
def test_missing_required_biome_input_retains_safe_fallback(missing_input: str) -> None:
    tensors: dict[str, object] = {
        "alive_mask": [True],
        "slot_to_agent_id": [101],
        "survival_steps": [3],
        "pos_y": [0],
        "pos_x": [1],
        "biome_grid": [[0, 1]],
    }
    tensors.pop(missing_input)

    assert _biome_fractions(tensors) == {
        "biome_safe_frac_alive": 1.0,
        "biome_wild_frac_alive": 0.0,
    }
