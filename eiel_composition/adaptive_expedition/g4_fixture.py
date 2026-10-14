"""Build the bounded G4 exact-branch engineering manifest fixture.

This is a profile-specific fixture builder, not coordinator truth or a new
contract.  Its output is the existing ``ExpeditionManifestV1`` JSON consumed
by ``eiel expedition run``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from emergence_contracts import canonical_json_bytes
from emergence_contracts.adaptive_science_expedition_v1 import (
    ExpeditionManifestV1,
    ObservationFrameV1,
    StateClassV1,
)
from emergence_contracts.artifacts import RunArtifactRefV1
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.science_request_v1 import validate_eiel_science_request_v1

from .general_closed_loop import EXACT_OBSERVATION_FRAME_ARTIFACT_KIND_V1
from .lifecycle import default_expedition_lifecycle_policy_value_v1
from .policies.adaptive_illumination_v1 import (
    ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_V1_PROFILE,
    ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_AXIS_ROUND_ROBIN_V1_PROFILE,
    AdaptiveIlluminationPolicyV1,
)
from .policy_protocol import canonical_plain_value_v1


G4_EXPEDITION_ID_V1 = "EIEL-ADAPTIVE-EXPEDITION-G4-TRUE-BRANCH-V1"
G4_OBSERVATION_PROFILE_ID_V1 = "EIEL_WORKHORSE_EXACT_ENGINEERING_OBSERVATION_V1"
G4_FOUNDER_NEXT_ABSOLUTE_TICK_V1 = 6
G4_FOUNDER_WORLD_UID_V1 = 210_000


class G4FixtureError(ValueError):
    """The engineering fixture cannot be frozen without weakening identity."""


def _read_object(path: Path, *, field: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise G4FixtureError(f"{field} must be a regular file")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise G4FixtureError(f"{field} is not readable JSON") from error
    if type(value) is not dict:
        raise G4FixtureError(f"{field} must contain an object")
    return value


def _file_uri_path(locator: str) -> Path:
    parsed = urlsplit(locator)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise G4FixtureError("exact founder frame locator must be a local file URI")
    return Path(unquote(parsed.path)).resolve(strict=True)


def build_g4_exact_manifest_v1(
    *,
    founder_request_path: str | Path,
    initial_frame_ref_path: str | Path | None = None,
    policy_profile: Mapping[str, object] = (
        ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_AXIS_ROUND_ROBIN_V1_PROFILE
    ),
    maximum_frontiers: int = 3,
    wall_time_budget_seconds: int = 1800,
) -> ExpeditionManifestV1:
    if type(maximum_frontiers) is not int or maximum_frontiers < 1:
        raise G4FixtureError("maximum_frontiers must be a positive integer")
    if type(wall_time_budget_seconds) is not int or wall_time_budget_seconds < 1:
        raise G4FixtureError("wall_time_budget_seconds must be positive")
    if not isinstance(policy_profile, Mapping):
        raise G4FixtureError("policy_profile must be a mapping")
    founder_path = Path(founder_request_path).expanduser().resolve(strict=True)
    founder = validate_eiel_science_request_v1(
        _read_object(founder_path, field="founder Science Request")
    )
    if founder.to_dict()["world"] != {
        "width": 64,
        "height": 64,
        "population": 1024,
        "capacity": 2048,
    } or founder.expect_worlds != 1:
        raise G4FixtureError("G4 founder declaration must be singleton W64")
    policy = AdaptiveIlluminationPolicyV1(policy_profile)
    descriptors = canonical_plain_value_v1(
        policy.profile["descriptor_definitions"]
    )
    descriptor_sha256 = canonical_sha256(descriptors)
    founder_bytes = canonical_json_bytes(founder.to_dict())
    refs: list[dict[str, object]] = [
        {
            "schema_version": "run_artifact_ref_v1",
            "artifact_id": f"founder-science-request-{founder.request_sha256().hex[:24]}",
            "artifact_kind": "eiel_science_request_v1",
            "artifact_schema_version": "eiel_science_request_v1",
            "locator": founder_path.as_uri(),
            "content_digest": founder.request_sha256().to_dict(),
            "size_bytes": len(founder_bytes),
        }
    ]
    if initial_frame_ref_path is not None:
        ref = RunArtifactRefV1.from_dict(
            _read_object(Path(initial_frame_ref_path), field="initial frame ref")
        )
        frame_path = _file_uri_path(ref.locator)
        frame_payload = frame_path.read_bytes()
        frame = ObservationFrameV1.from_dict(json.loads(frame_payload))
        if (
            ref.artifact_kind != EXACT_OBSERVATION_FRAME_ARTIFACT_KIND_V1
            or str(ref.artifact_schema_version)
            != EXACT_OBSERVATION_FRAME_ARTIFACT_KIND_V1
            or frame_payload != canonical_json_bytes(frame.to_dict())
            or ref.content_digest != frame.observation_frame_sha256()
            or ref.size_bytes != len(frame_payload)
            or frame.expedition_id != G4_EXPEDITION_ID_V1
            or frame.observation_profile_id != G4_OBSERVATION_PROFILE_ID_V1
            or frame.observation_descriptor_set_sha256 != descriptor_sha256
            or frame.logical_world_id != G4_FOUNDER_WORLD_UID_V1
            or frame.next_absolute_tick != G4_FOUNDER_NEXT_ABSOLUTE_TICK_V1
            or frame.source_state_ref.state_class is not StateClassV1.BRANCHABLE_EXACT
        ):
            raise G4FixtureError("initial exact frame differs from the frozen G4 basis")
        refs.append(ref.to_dict())
    refs.sort(key=lambda row: str(row["artifact_id"]))
    outer_seed = int(
        canonical_sha256(
            {
                "expedition_id": G4_EXPEDITION_ID_V1,
                "founder_request_sha256": founder.request_sha256().to_dict(),
                "policy_binding": canonical_plain_value_v1(policy.binding),
            }
        ).hex[:16],
        16,
    )
    manifest = ExpeditionManifestV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_expedition_manifest_v1",
            "expedition_id": G4_EXPEDITION_ID_V1,
            "question_description": (
                "Bounded no-claims exact selected-state engineering expedition over "
                "resource supply and mutation rate"
            ),
            "no_claims": True,
            "hypothesis_ids": [],
            "required_science_instrument": "EIEL_SCIENCE_INSTRUMENT_V1",
            "founder_basis_refs": refs,
            "search_space": {
                "schema_version": "eiel_adaptive_expedition_search_space_v1",
                "payload": {
                    "controls": [
                        {
                            "control_id": "environment.resource.multiplier",
                            "allowed_values": [1.0, 1.5],
                        },
                        {
                            "control_id": "mutation.rate",
                            "allowed_values": [0.125, 0.25],
                        },
                    ]
                },
            },
            "epoch_ticks": 2,
            "maximum_absolute_tick": 12,
            "maximum_frontiers": maximum_frontiers,
            "maximum_actions_per_frontier": 1,
            "wall_time_budget_seconds": wall_time_budget_seconds,
            "observation_profile_id": G4_OBSERVATION_PROFILE_ID_V1,
            "observation_descriptor_set_sha256": descriptor_sha256.to_dict(),
            "policy_binding": canonical_plain_value_v1(policy.binding),
            "outer_search_seed": outer_seed,
            "maximum_atlas_entries": 64,
            # Keep the fixture's manifest budget aligned with the bound
            # profile.  The profile may still reject candidates for its
            # divergence/priority rules; the coordinator never truncates a
            # policy-owned bank against a smaller hidden capacity.
            "maximum_retained_parent_entries": 32,
            "maximum_retained_parent_bytes": 64 * 1024 * 1024,
            "artifact_lifecycle_policy": {
                "schema_version": "eiel_adaptive_expedition_lifecycle_policy_v1",
                "payload": default_expedition_lifecycle_policy_value_v1(),
            },
            "stop_conditions": [
                "integrity_failure",
                "operator_stop",
                "resource_budget_exhausted",
            ],
        }
    )
    if canonical_plain_value_v1(manifest.policy_binding.to_dict()) != (
        canonical_plain_value_v1(policy.binding)
    ):
        raise G4FixtureError("installed exact policy differs from the manifest binding")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--founder-request", required=True)
    parser.add_argument("--initial-frame-ref")
    parser.add_argument(
        "--policy-profile",
        choices=("axis-round-robin", "exact-workhorse"),
        default="axis-round-robin",
    )
    parser.add_argument("--maximum-frontiers", type=int, default=3)
    parser.add_argument("--wall-time-budget-seconds", type=int, default=1800)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    profile = (
        ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_V1_PROFILE
        if args.policy_profile == "exact-workhorse"
        else ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_AXIS_ROUND_ROBIN_V1_PROFILE
    )
    manifest = build_g4_exact_manifest_v1(
        founder_request_path=args.founder_request,
        initial_frame_ref_path=args.initial_frame_ref,
        policy_profile=profile,
        maximum_frontiers=args.maximum_frontiers,
        wall_time_budget_seconds=args.wall_time_budget_seconds,
    )
    output = Path(args.output).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(manifest.to_dict())
    with output.open("xb") as handle:
        handle.write(payload)
    print(
        json.dumps(
            {
                "status": (
                    "G4_EXACT_MANIFEST_FINAL"
                    if args.initial_frame_ref is not None
                    else "G4_EXACT_MANIFEST_DRAFT_REQUIRES_INITIAL_FRAME"
                ),
                "manifest_path": str(output.resolve()),
                "manifest_sha256": manifest.manifest_sha256().hex,
                "founder_basis_ref_count": len(manifest.founder_basis_refs),
                "execution_started": False,
                "qualification_authority_effect": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "G4FixtureError",
    "G4_EXPEDITION_ID_V1",
    "G4_FOUNDER_NEXT_ABSOLUTE_TICK_V1",
    "G4_FOUNDER_WORLD_UID_V1",
    "G4_OBSERVATION_PROFILE_ID_V1",
    "build_g4_exact_manifest_v1",
)
