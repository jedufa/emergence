#!/usr/bin/env python3
from __future__ import annotations

"""Build HABACCLIM-0R gate evidence up to the first failed gate."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from env.gridworld_pkg.patch_world import PATCH_WORLD_OBSERVATION_CHANNEL_NAMES


EXPECTED_HC1_ZIP_SHA256 = "daaf03aac12e3549c3c6efe1c5c18df155b50ccc7335db6fac15e553405ee4e2"
EXPECTED_HC1_CONSTRUCT_SHA256 = "09f0762db8acecef87ffb22a6a547a11417f57cfc6763e083443502b6904fa6b"
REJECTED_PROMPT_ONLY_HC1_SHA256 = "70b45965968532b4507053ef410e4a947e372adcf9ccb34e0b6d5b86534ba010"
PRIOR_HABACCLIM0_STOP_ZIP_SHA256 = "0c41c66eb4811fbce9500862f9979eca6c778f0bec711f4f818046277444a32e"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ordinary_observation_fixture() -> dict[str, Any]:
    channels = ("legacy_food_occupancy", *PATCH_WORLD_OBSERVATION_CHANNEL_NAMES[1:])
    obs = np.zeros((74,), dtype=np.float32)
    cells = {
        "center": 4,
        "north": 1,
        "east": 5,
        "south": 7,
        "west": 3,
    }
    for cell, idx in cells.items():
        obs[channels.index("patch_food_amount_norm") * 9 + idx] = np.float32(0.82 if cell == "center" else 0.78)
        obs[channels.index("patch_cover") * 9 + idx] = np.float32(0.62)
        obs[channels.index("patch_shelter") * 9 + idx] = np.float32(0.55)
        obs[channels.index("patch_movement_cost_norm") * 9 + idx] = np.float32(0.30)
    obs[72] = np.float32(0.70)
    obs[73] = np.float32(1.00)
    a_acclimated_obs = obs.copy()
    b_acclimated_obs = obs.copy()
    return {
        "schema_version": "habacclim0r_observation_sufficiency_fixture_v1",
        "fixture": "same_habitat_same_local_cues_energy_health_opposite_acclimation",
        "ordinary_observation_channels": list(channels),
        "ordinary_scalar_tail": ["energy_norm", "health_norm"],
        "candidate_required_but_absent_scalar": "habitat_acclimation_state_norm",
        "a_acclimated_state": 1.0,
        "b_acclimated_state": -1.0,
        "ordinary_observations_identical_without_acclimation_scalar": bool(
            np.array_equal(a_acclimated_obs, b_acclimated_obs)
        ),
        "a_acclimated_required_behavior": "gather_or_stay_in_habitat_a",
        "b_acclimated_required_behavior": "rest_or_depart_due_to_a_mismatch_cost",
        "required_behaviors_differ": True,
        "reason": (
            "The acquired state is the commitment mechanism. Without an ordinary normalized acclimation scalar, "
            "a deterministic memoryless MLP receives identical input for matched and mismatched agents in the same "
            "habitat at the same energy and health."
        ),
        "obs_sha256": hashlib.sha256(a_acclimated_obs.tobytes()).hexdigest(),
    }


def build_payload(root: Path) -> dict[str, Any]:
    hc1_zip = root / "evidences/07-world-pressure/23-corridor-hc1-homing-cue-final-validation.evidence.zip"
    hc1_record = (
        root
        / "evidences/07-world-pressure/23-corridor-hc1-homing-cue-final-validation/corridor_hc1_probe_construct_search_record.json"
    )
    prior_stop = root / "evidences/07-world-pressure/24-habacclim0-commitment-cost-validation.evidence.zip"
    actual_hc1_zip = _sha256(hc1_zip)
    actual_hc1_record = _sha256(hc1_record)
    actual_prior_stop = _sha256(prior_stop)
    fixture = _ordinary_observation_fixture()
    return {
        "schema_version": "habacclim0r_gate_audit_v1",
        "decision": "HABACCLIM-0R-E_OBSERVATION_SUFFICIENCY_FAILURE",
        "scientific_cells_run": 0,
        "source_repair": {
            "hc1_zip_expected": EXPECTED_HC1_ZIP_SHA256,
            "hc1_zip_actual": actual_hc1_zip,
            "hc1_zip_ok": actual_hc1_zip == EXPECTED_HC1_ZIP_SHA256,
            "hc1_construct_expected": EXPECTED_HC1_CONSTRUCT_SHA256,
            "hc1_construct_actual": actual_hc1_record,
            "hc1_construct_ok": actual_hc1_record == EXPECTED_HC1_CONSTRUCT_SHA256,
            "rejected_prompt_only_hc1_sha256": REJECTED_PROMPT_ONLY_HC1_SHA256,
            "prior_habacclim0_stop_expected": PRIOR_HABACCLIM0_STOP_ZIP_SHA256,
            "prior_habacclim0_stop_actual": actual_prior_stop,
            "prior_habacclim0_stop_ok": actual_prior_stop == PRIOR_HABACCLIM0_STOP_ZIP_SHA256,
        },
        "gates": {
            "1_corridor_closure_integrity": "PASS",
            "2_scientific_target_clarity": "PASS",
            "3_acquired_acclimation_legitimacy": "PASS_DESIGN_ONLY",
            "4_ecological_demand": "PASS_DESIGN_ONLY",
            "5_action_expressibility": "PASS",
            "6_cue_and_observation_sufficiency": "FAIL_OBSERVATION_CONTRACT",
            "7_ordinary_policy_representation": "NOT_EVALUATED",
            "8_probe_construct_validity": "NOT_EVALUATED",
            "9_mutation_and_inheritance_relevance": "NOT_EVALUATED",
            "10_incompatible_ecological_cost": "NOT_EVALUATED",
            "11_analytic_reproductive_consequence": "NOT_EVALUATED",
            "12_anti_generalist_argument": "NOT_EVALUATED",
            "13_controls": "NOT_EVALUATED",
            "14_measurement": "NOT_EVALUATED",
            "15_timing_overlap": "NOT_EVALUATED",
            "16_implementation_scope": "NOT_EVALUATED",
            "17_integrity": "NOT_EVALUATED",
            "18_registry_readiness": "NOT_EVALUATED",
        },
        "source_symbols": {
            "ordinary_observation_api": "env.gridworld_pkg.observations.ObservationsMixin.get_local_observations",
            "ordinary_patch_channels": "env.gridworld_pkg.patch_world.PATCH_WORLD_OBSERVATION_CHANNEL_NAMES",
            "fixed_mlp_probe_obs_tail": [
                "simulation.predrisk_fixed1_probe._obs_batch obs[72]=energy_norm",
                "simulation.predrisk_fixed1_probe._obs_batch obs[73]=health_norm",
                "simulation.corridor_cp0_fixed_probe._obs_batch obs[72]=energy_norm",
                "simulation.corridor_cp0_fixed_probe._obs_batch obs[73]=health_norm",
            ],
        },
        "gate_6_failure": fixture,
        "bounded_interpretation": (
            "HABACCLIM-0R is stopped before implementation because the required acquired physiological state is not "
            "part of the current ordinary observation contract. Adding it may be a future prerequisite, but running "
            "fixed probes now would either hide the commitment state from policy or expose it through a nonordinary shortcut."
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="experiments/projects/emergence root")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    out = Path(args.out_dir).resolve()
    payload = build_payload(root)
    _write_json(out / "habacclim0r_gate_audit.json", payload)
    _write_json(out / "habacclim0r_source_repair_verification.json", payload["source_repair"])
    _write_json(out / "habacclim0r_observation_sufficiency_fixture.json", payload["gate_6_failure"])
    _write_json(
        out / "temp_c3_dossier.json",
        {
            "schema_version": "habacclim0r_design_only_challenger_dossier_v1",
            "challenger": "TEMP-C3",
            "status": "DESIGN_ONLY_NOT_IMPLEMENTED_NOT_SIMULATED",
            "anti_generalist_mechanism_required": "A genotype-blind commitment state must impose lag without exposing phase labels or prediction targets.",
            "ordinary_policy_representation_risk": "If a temporal cue directly labels phase, one universal conditional policy may dominate.",
            "controls_required": ["true_flat", "labels_only", "zero_lag", "uncorrelated_temporal_cost", "budget_matched_uniform"],
            "first_failed_gate": "not_evaluated_after_habacclim0r_gate_6_stop",
        },
    )
    _write_json(
        out / "stagehab_c2_dossier.json",
        {
            "schema_version": "habacclim0r_design_only_challenger_dossier_v1",
            "challenger": "STAGEHAB-C2",
            "status": "DESIGN_ONLY_NOT_IMPLEMENTED_NOT_SIMULATED",
            "anti_generalist_mechanism_required": "Juvenile-safe/adult-rich habitats must not collapse into one universal age-switching policy.",
            "ordinary_policy_representation_risk": "Direct life-stage observation plus direct habitat labels can eliminate inherited strategy contrast.",
            "controls_required": ["true_flat", "stage_labels_only", "food_contrast_only", "zero_stage_cost", "budget_matched_uniform"],
            "first_failed_gate": "not_evaluated_after_habacclim0r_gate_6_stop",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
