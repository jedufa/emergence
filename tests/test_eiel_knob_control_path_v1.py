from __future__ import annotations

import pytest

from engine.env.env_config import EnvConfig
from engine.evolution.population_pkg.population_config import PopConfig
from engine.knobs_override import (
    DEAD_PRODUCTION_KNOB_FIELDS,
    apply_knob_overrides,
    verify_knob_override_readback,
)
from engine.simulation.runner_pkg.runner_config import SimConfig


@pytest.mark.parametrize(
    "field_name",
    sorted(DEAD_PRODUCTION_KNOB_FIELDS["env_cfg"]),
)
def test_generic_knob_rejects_confirmed_dead_behavior_fields(
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match="has no runtime consumer"):
        apply_knob_overrides(
            knob_pairs=[f"env_cfg.{field_name}=true"],
            env_cfg=EnvConfig(),
            pop_cfg=PopConfig(),
            sim_cfg=SimConfig(),
        )


def test_generic_knob_final_readback_matches_requested_value() -> None:
    env_cfg = EnvConfig()
    pop_cfg = PopConfig()
    sim_cfg = SimConfig()

    applied, _genome = apply_knob_overrides(
        knob_pairs=["env_cfg.patch_food_regrow_per_step=0.125"],
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
    )

    receipts = verify_knob_override_readback(
        applied_overrides=applied,
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
    )

    assert receipts == [
        {
            "schema_version": "eiel_generic_knob_readback_v1",
            "control": "env_cfg.patch_food_regrow_per_step",
            "requested": 0.125,
            "readback": 0.125,
            "status": "FINAL_CONFIG_READBACK_COMPLETE",
            "mechanic_connectivity_qualified": False,
            "wrc_admitted": False,
            "qualification_note": (
                "generic knob readback proves final dataclass state only; "
                "WRC requires a typed production-route control receipt"
            ),
        }
    ]


def test_generic_knob_fails_closed_when_later_owner_overwrites_value() -> None:
    env_cfg = EnvConfig()
    pop_cfg = PopConfig()
    sim_cfg = SimConfig()

    applied, _genome = apply_knob_overrides(
        knob_pairs=["env_cfg.reproduce_try_prob=0.25"],
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
    )
    env_cfg.reproduce_try_prob = 1.0

    with pytest.raises(ValueError, match="KNOB_OVERRIDDEN"):
        verify_knob_override_readback(
            applied_overrides=applied,
            env_cfg=env_cfg,
            pop_cfg=pop_cfg,
            sim_cfg=sim_cfg,
        )
