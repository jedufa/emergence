from __future__ import annotations

"""simulation/runner_pkg/runner_base.py

Shared runner state + constructor plumbing.

This module is intentionally lightweight: it stores configuration, constructs the
environment + population, and owns long-lived runner state used by mixins.

Internal drives (performance refactor)
--------------------------------------
Drives are hot-path state used every tick for every alive agent. Storing them
inside Python Agent objects forces per-agent Python overhead. We therefore keep
authoritative drives as dense per-slot arrays on the runner:

    self._drive_hunger, _drive_pain, _drive_fatigue, _drive_fear

These are indexed by agent.slot (not agent_id). We track slot reuse via a
parallel int32 token array self._drive_slot_token and compare against the
agent's slot_token.

Agents still keep an `Agent.drives` vector as a compatibility mirror for any
code paths that read it.
"""

from typing import Any, Dict, Optional
import copy
import hashlib
import json
import os
import numpy as np
import torch

from agents.batched_policy import forward_batched_mlp  # noqa: F401 (kept for legacy imports)
from agents.chemistry import CHEMISTRY_SIZE, chemistry_from_values, zeros_chemistry
from agents.maintenance import MAINTENANCE_SIZE, maintenance_from_values, zeros_maintenance_state
from agents.developmental_program_v1 import apply_irreversible_stage_locks, canonicalize_developmental_program
from agents.modular_genome_v1 import (
    B2_FROZEN_CAPS,
    B3_FROZEN_LOCALITY,
    resolve_b2_caps,
    strict_mode_enabled as _b2_strict_enabled,
)
from agents.structural_identity_v1 import (
    STRUCTURAL_ANCESTRY_SCHEMA_VERSION,
    STRUCTURAL_IDENTITY_SCHEMA_VERSION,
    TOPOLOGY_POSITION_SCHEMA_VERSION,
)
from evolution.operators_c1 import (
    C1_FROZEN_INVALID_OPERATOR_POLICY,
    C1_FROZEN_MAX_OPS_PER_OFFSPRING,
    C1_OPERATOR_CATALOG_VERSION,
)
from evolution.recombination_c2 import (
    C2_FROZEN_ALIGNMENT_KEY,
    C2_FROZEN_UNMATCHED_POLICY,
    C2_LEGACY_UNRESOLVED_POLICY,
    C2_RECOMBINATION_VERSION,
)
from evolution.structural_live_semantics import (
    apply_live_structural_semantics,
    structural_live_semantics_enabled,
)
from evolution.structural_live_modules_v1 import (
    LIVE_MODULE_SCHEMA_VERSION,
    LIVE_PROFILE,
    validate_live_modules_config,
)
from evolution.structural_observation_routing_v1 import validate_routing_config
from evolution.structural_output_routing_v1 import validate_output_routing_config
from evolution.structural_development_v1 import (
    structural_development_enabled,
    validate_structural_development_config,
)
from evolution.structural_gated_recurrent_v1 import (
    GATED_C2_COMPATIBILITY_VERSION,
    GATED_DRAW_DECOMPOSITION_VERSION,
    GATED_DUPLICATION_POLICY_VERSION,
    GATED_EQUATION_VERSION,
    GATED_FUSION_POLICY_VERSION,
    GATED_LATENT_STATE_POLICY_VERSION,
    GATED_MUTATION_BUDGET_VERSION,
    GATED_MUTATION_PROFILE,
    GATED_NEUTRAL_ELIGIBILITY_VERSION,
    GATED_NUMERIC_VERSION,
    GATED_OPERATION_BUDGET_VERSION,
    GATED_OPERATOR,
    GATED_PROFILE,
    GATED_PROMOTER_POLICY_VERSION,
    GATED_RECORD_SCHEMA_VERSION,
    GATED_RUNTIME_SCHEMA_VERSION,
    GATED_STATE_WIDTH,
    GATED_SUPPORTED_LANE,
    escap7_unsupported_lane_reasons,
)
from evolution.structural_sustainable_growth_v1 import (
    ACTIVITY_SCHEMA_VERSION,
    BLOAT_CLASSIFICATION_VERSION,
    C2_COMPATIBILITY_VERSION as SUSTAINABLE_GROWTH_C2_COMPATIBILITY_VERSION,
    CHECKPOINT_SCHEMA_VERSION as SUSTAINABLE_GROWTH_CHECKPOINT_SCHEMA_VERSION,
    COST_COEFFICIENTS,
    COST_MODE,
    COST_PROFILE,
    COST_SCHEMA_VERSION,
    DUPLICATION_NEUTRALIZATION_VERSION,
    FUSION_POLICY_VERSION as SUSTAINABLE_GROWTH_FUSION_POLICY_VERSION,
    GROWTH_MUTATION_ATTEMPTS,
    GROWTH_MUTATION_BINS,
    GROWTH_MUTATION_DRAWS,
    GROWTH_MUTATION_PROFILE,
    MIGRATION_RNG_DRAWS as SUSTAINABLE_GROWTH_MIGRATION_RNG_DRAWS,
    PRUNE_CLEANUP_VERSION,
    PRUNE_ELIGIBILITY_VERSION,
    REPACK_VERSION,
    REFINEMENT_OPERATOR_VERSION,
    SPECIALIZATION_TELEMETRY_VERSION,
    SUPPORTED_LANE as SUSTAINABLE_GROWTH_SUPPORTED_LANE,
    SUSTAINABLE_GROWTH_PROFILE,
)
from env.gridworld import GridWorld
from env.env_config import EnvConfig
from env.gridworld_pkg.patch_world import (
    patch_world_local_observation_size,
    patch_world_total_observation_size,
)
from evolution.population import Population, PopConfig
from evolution.inheritance_registry_v1 import (
    validate_inheritance_execution_lane,
    validate_production_inheritance_profile,
)
from simulation.metrics import MetricsLogger, BehaviorLogger
from simulation.metrics_pkg.noop_loggers import NoOpBehaviorLogger
from simulation.runner_pkg.perf_writer import PerfBucketWriter
from simulation.rng_manager import RNGManager
from simulation.control_plane_pkg import ControlPlane
from simulation.autopilot_pkg import PressureAutopilot
from simulation.memory_action_window import MemoryActionWindowInstrumentation
from simulation.mem_max_live_level4 import MemMaxLiveLevel4Instrumentation
from simulation.mem_max_pre_cue_checkpoint import MemMaxPreCueCheckpointRecorder
from simulation.prediction_demand_action_context import PredictionDemandActionContextInstrumentation
from perf.cxxcore23_live_direct_compact_diagnostics import LiveDirectCompactDiagnostics
from simulation.env_hotpath_expression_diagnostics import EnvHotpathExpressionDiagnostics
from simulation.branch_observability import BranchObservabilityWriter
from simulation.chronicle import ScientificChronicleRecorder
from simulation.chronicle.gen1_f1 import Gen1F1Capture


def _resolve_torch_device(requested: str) -> torch.device:
    dev_str = str(requested).lower()
    if dev_str in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev_str == "cpu":
        return torch.device("cpu")
    if dev_str == "gpu":
        dev_str = "cuda"
    if dev_str.startswith("cuda"):
        try:
            device = torch.device(dev_str)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Malformed explicit CUDA device request: {requested!r}") from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Explicit CUDA device request {requested!r} cannot be satisfied: CUDA is unavailable"
            )

        device_count = int(torch.cuda.device_count())
        if device_count <= 0:
            raise RuntimeError(
                f"Explicit CUDA device request {requested!r} cannot be satisfied: no CUDA devices are visible"
            )
        if device.index is not None and device.index >= device_count:
            raise RuntimeError(
                f"Explicit CUDA device request {requested!r} is out of range: "
                f"{device_count} CUDA device(s) are visible"
            )
        return device
    raise ValueError(f"Unsupported device request: {requested!r}")


_ESCAP7_FIXED_CONFIG = {
    "structural_gated_recurrent_profile": GATED_PROFILE,
    "structural_gated_recurrent_schema_version": GATED_RUNTIME_SCHEMA_VERSION,
    "structural_gated_recurrent_operator": GATED_OPERATOR,
    "structural_gated_recurrent_parameter_schema_version": GATED_RECORD_SCHEMA_VERSION,
    "structural_gated_recurrent_equation_version": GATED_EQUATION_VERSION,
    "structural_gated_recurrent_numeric_policy_version": GATED_NUMERIC_VERSION,
    "structural_gated_recurrent_state_width": GATED_STATE_WIDTH,
    "structural_gated_recurrent_neutral_eligibility_version": GATED_NEUTRAL_ELIGIBILITY_VERSION,
    "structural_gated_recurrent_neutral_latent_state_policy": GATED_LATENT_STATE_POLICY_VERSION,
    "structural_gated_recurrent_mutation_operator_profile": GATED_MUTATION_PROFILE,
    "structural_gated_recurrent_operator_attempts": 1,
    "structural_gated_recurrent_operator_draws_per_attempt": 2,
    "structural_gated_recurrent_draw_bin_count": 5,
    "structural_gated_recurrent_decomposition_version": GATED_DRAW_DECOMPOSITION_VERSION,
    "structural_gated_recurrent_mutation_budget_version": GATED_MUTATION_BUDGET_VERSION,
    "structural_gated_recurrent_operation_budget_version": GATED_OPERATION_BUDGET_VERSION,
    "structural_gated_recurrent_duplication_policy_version": GATED_DUPLICATION_POLICY_VERSION,
    "structural_gated_recurrent_promoter_policy_version": GATED_PROMOTER_POLICY_VERSION,
    "structural_gated_recurrent_fusion_policy_version": GATED_FUSION_POLICY_VERSION,
    "structural_gated_recurrent_c2_compatibility_version": GATED_C2_COMPATIBILITY_VERSION,
    "structural_gated_recurrent_supported_lane": GATED_SUPPORTED_LANE,
    "structural_gated_recurrent_migration_rng_draws": 0,
}


_ESCAP8_FIXED_CONFIG = {
    "structural_sustainable_growth_profile": SUSTAINABLE_GROWTH_PROFILE,
    "structural_sustainable_growth_checkpoint_schema_version": SUSTAINABLE_GROWTH_CHECKPOINT_SCHEMA_VERSION,
    "structural_sustainable_growth_activity_schema_version": ACTIVITY_SCHEMA_VERSION,
    "structural_sustainable_growth_duplication_neutralization_version": DUPLICATION_NEUTRALIZATION_VERSION,
    "structural_sustainable_growth_refinement_operator_version": REFINEMENT_OPERATOR_VERSION,
    "structural_sustainable_growth_specialization_telemetry_version": SPECIALIZATION_TELEMETRY_VERSION,
    "structural_sustainable_growth_prune_eligibility_version": PRUNE_ELIGIBILITY_VERSION,
    "structural_sustainable_growth_prune_cleanup_version": PRUNE_CLEANUP_VERSION,
    "structural_sustainable_growth_repack_version": REPACK_VERSION,
    "structural_sustainable_growth_fusion_policy_version": SUSTAINABLE_GROWTH_FUSION_POLICY_VERSION,
    "structural_sustainable_growth_cost_schema_version": COST_SCHEMA_VERSION,
    "structural_sustainable_growth_cost_profile": COST_PROFILE,
    "structural_sustainable_growth_cost_mode": COST_MODE,
    "structural_sustainable_growth_cost_coefficients": {
        str(key): float(value) for key, value in COST_COEFFICIENTS.items()
    },
    "structural_sustainable_growth_bloat_classification_version": BLOAT_CLASSIFICATION_VERSION,
    "structural_sustainable_growth_operator_profile": GROWTH_MUTATION_PROFILE,
    "structural_sustainable_growth_operator_attempts": GROWTH_MUTATION_ATTEMPTS,
    "structural_sustainable_growth_operator_draws_per_attempt": GROWTH_MUTATION_DRAWS,
    "structural_sustainable_growth_c1_lane_map": tuple(GROWTH_MUTATION_BINS),
    "structural_sustainable_growth_c2_compatibility_version": SUSTAINABLE_GROWTH_C2_COMPATIBILITY_VERSION,
    "structural_sustainable_growth_supported_lane": SUSTAINABLE_GROWTH_SUPPORTED_LANE,
    "structural_sustainable_growth_migration_rng_draws": SUSTAINABLE_GROWTH_MIGRATION_RNG_DRAWS,
}


def _validate_escap7_gated_recurrent_config(
    cfg: Any,
    *,
    policy_device: torch.device,
    bank_device: torch.device,
) -> None:
    """Fail closed before ticks when the ESCAP-7 reference lane is requested."""

    if not bool(getattr(cfg, "structural_gated_recurrent_enable", False)):
        return

    mismatches = {
        name: {"expected": expected, "actual": getattr(cfg, name, None)}
        for name, expected in _ESCAP7_FIXED_CONFIG.items()
        if getattr(cfg, name, None) != expected
    }
    if mismatches:
        raise RuntimeError(f"ESCAP7_CONFIG_MISMATCH fields={mismatches!r}")
    live_prerequisite = (
        bool(getattr(cfg, "structural_live_modules_enable", False)),
        str(getattr(cfg, "structural_live_modules_profile", "none") or "none"),
        str(
            getattr(cfg, "structural_live_modules_schema_version", LIVE_MODULE_SCHEMA_VERSION)
            or LIVE_MODULE_SCHEMA_VERSION
        ),
    )
    if live_prerequisite != (True, LIVE_PROFILE, LIVE_MODULE_SCHEMA_VERSION):
        raise RuntimeError(
            "ESCAP7_LIVE_MODULE_PREREQUISITE_MISMATCH "
            f"expected={(True, LIVE_PROFILE, LIVE_MODULE_SCHEMA_VERSION)!r} "
            f"actual={live_prerequisite!r}"
        )

    lane_reasons = escap7_unsupported_lane_reasons(
        cfg,
        policy_device=policy_device,
        bank_device=bank_device,
    )

    if lane_reasons:
        raise RuntimeError(
            "ESCAP7_UNSUPPORTED_EXECUTION_LANE "
            f"reasons={sorted(set(lane_reasons))!r}"
        )


def _validate_escap8_sustainable_growth_config(
    cfg: Any,
    *,
    policy_device: torch.device,
    bank_device: torch.device,
) -> None:
    """Fail closed before ticks for the enabled ESCAP-8 reference lane."""

    if not bool(getattr(cfg, "structural_sustainable_growth_enable", False)):
        return
    if not bool(getattr(cfg, "structural_gated_recurrent_enable", False)):
        raise RuntimeError("ESCAP8_ESCAP7_PREREQUISITE_MISMATCH enabled=0")
    mismatches = {
        name: {"expected": expected, "actual": getattr(cfg, name, None)}
        for name, expected in _ESCAP8_FIXED_CONFIG.items()
        if getattr(cfg, name, None) != expected
    }
    if mismatches:
        raise RuntimeError(f"ESCAP8_CONFIG_MISMATCH fields={mismatches!r}")
    lane_reasons = escap7_unsupported_lane_reasons(
        cfg,
        policy_device=policy_device,
        bank_device=bank_device,
    )
    if lane_reasons:
        raise RuntimeError(
            "ESCAP8_UNSUPPORTED_EXECUTION_LANE "
            f"reasons={sorted(set(lane_reasons))!r}"
        )


class RunnerBase:
    """Shared state + constructor. Split from heavy loop logic."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        pop_cfg: PopConfig,
        sim_cfg,
        logger: MetricsLogger,
        run_dir: str,
        perf: bool = False,
    ):
        self.env_cfg = env_cfg
        self.pop_cfg = pop_cfg
        self.sim_cfg = sim_cfg
        self.logger = logger
        self.run_dir = run_dir
        self.perf = bool(perf)
        # Fail closed before constructing RNGManager (or any stochastic
        # subsystem).  Validation consumes zero draws.
        self.sim_cfg.inheritance_profile = validate_production_inheritance_profile(
            getattr(self.sim_cfg, "inheritance_profile", None)
        )
        self._enforce_b2_campaign_freeze()
        self.substrate_capacity_manager = None
        if bool(getattr(self.sim_cfg, "capacity_epoch_manager_enable", False)):
            from simulation.capacity_epoch import SubstrateCapacityManager  # noqa: WPS433

            if str(getattr(self.sim_cfg, "device", "cpu") or "cpu").lower() != "cpu":
                raise RuntimeError("substrate capacity manager requires explicit CPU device")
            self.sim_cfg.capacity_epoch_metadata_enable = True
            self.substrate_capacity_manager = SubstrateCapacityManager()

        # -----------------------------
        # Phase 5d: viz/ observability artifacts (UI read-only helpers)
        # -----------------------------
        # These files live under <RUN_DIR>/viz/ and are explicitly NOT part of
        # the determinism substrate.
        try:
            from simulation.viz_pkg.viz_writer import get_viz_writer
            self.viz = get_viz_writer(self.run_dir)
        except Exception:
            self.viz = None

        # Optional: persist perf buckets to CSV for later comparison
        self._perf_writer = PerfBucketWriter(run_dir=self.run_dir) if self.perf else None

        if hasattr(self.logger, "write_run_config"):
            self.logger.write_run_config(env_cfg, pop_cfg, sim_cfg)

        # -----------------------------
        # Phase 1: RNGManager authority
        # -----------------------------
        # All randomness must be routed through RNGManager (no RNG islands).
        self.rng_manager = RNGManager(global_seed=int(getattr(sim_cfg, "seed", 0) or 0))

        # -----------------------------
        # Phase 2: ControlPlane (runtime interventions + WAL)
        # -----------------------------
        # Minimal: branch_id/campaign_id are optional fields on sim_cfg.
        # If absent, we fall back to deterministic defaults.
        try:
            _branch_id = str(getattr(self.sim_cfg, "branch_id", "default") or "default")
            _campaign_id = str(getattr(self.sim_cfg, "campaign_id", "") or "")
        except Exception:
            _branch_id = "default"
            _campaign_id = ""
        self.control_plane = ControlPlane(run_dir=self.run_dir, branch_id=_branch_id, campaign_id=_campaign_id, initial_generation=0)
        # Expose submission sequence number for checkpoint metadata integration.
        self._control_plane_submission_sequence_number = int(self.control_plane.get_submission_sequence_number())

        # -----------------------------
        # Phase 6 (draft): Pressure Autopilot (per-run controller)
        # -----------------------------
        # This controller is OFF by default (sim_cfg.pressure_autopilot_enable=false).
        # When enabled, it schedules PressureUpdateCommand interventions via ControlPlane.
        self.pressure_autopilot = PressureAutopilot(run_dir=self.run_dir)


        # Pass run_dir so the env can log static world artifacts (e.g. Tier 1 spatial ecology maps).
        self.env = GridWorld(env_cfg, seed=sim_cfg.seed, run_dir=run_dir, rng=self.rng_manager.env_np)

        # Tier 4 sanity line (prints once at run start when enabled)
        if bool(getattr(env_cfg, "tier4_juvenile_vulnerability_enable", False)):
            print(
                "[tier4] juvenile vulnerability enabled: juvenile_steps=%s grace_steps=%s hazard_mult=%s move_penalty=%s harvest_mult=%s"
                % (
                    int(getattr(env_cfg, "tier4_juvenile_steps", 0) or 0),
                    int(getattr(env_cfg, "tier4_juvenile_grace_steps", 0) or 0),
                    float(getattr(env_cfg, "tier4_juvenile_hazard_multiplier", 1.0)),
                    float(getattr(env_cfg, "tier4_juvenile_movement_penalty", 1.0)),
                    float(getattr(env_cfg, "tier4_juvenile_harvest_multiplier", 1.0)),
                )
            )
        self.pop = Population(
            env_cfg,
            pop_cfg,
            seed=sim_cfg.seed,
            rng=self.rng_manager.evolution_py,
            np_rng=self.rng_manager.evolution_np,
        )

        # Phase 1.5: allow Population mixins (spawn/repro) to access sim_cfg if present.
        # This keeps backward compatibility: Population does not *require* sim_cfg.
        try:
            setattr(self.pop, "sim_cfg", sim_cfg)
        except Exception:
            pass
        try:
            setattr(self.pop, "rng_manager", self.rng_manager)
        except Exception:
            pass
        try:
            setattr(self.pop, "env", self.env)
        except Exception:
            pass
        # -----------------------------
        # Phase 5B: Trait Injection integration (plumbing only)
        # -----------------------------
        # Population-level trait injection hooks need access to the run directory
        # so they can read the ControlPlane event log at:
        #   <run_dir>/events/command_events.jsonl
        #
        # This is best-effort and does not affect determinism unless you actually
        # schedule TraitInjectionCommand interventions.
        try:
            setattr(self.pop, "run_dir", str(self.run_dir))
        except Exception:
            pass

        # Runner-level numpy RNG for misc non-env/non-evolution sampling.
        self.np_rng = self.rng_manager.metrics_np

        # -----------------------------
        # Multi-tick policy batching cache
        # -----------------------------
        # Populated on-demand by TickLoopMixin when policy_tick_stride > 1.
        # Keyed by agent slot with slot_token guards to avoid stale actions.
        self._policy_action_cache = {}
        self._dev_expr_cache: Dict[tuple[int, int, int, int], Dict[str, Any]] = {}
        self._dev_effective_scalar_cache: Dict[tuple[int, int, int, int], tuple[float, ...]] = {}
        self._dev_trace_pending: Dict[tuple[int, int], Dict[int, Dict[str, Any]]] = {}
        self._dev_lock_trace_pending: Dict[tuple[int, int], Dict[int, Dict[str, Any]]] = {}
        self._dev_trace_emit_buffer: list[str] = []
        self._dev_lock_trace_emit_buffer: list[str] = []
        self._dev_trace_emit_dirs_ready: set[str] = set()
        self._dev_trace_emit_counters: Dict[str, int] = {
            "records_queued": 0,
            "records_written": 0,
            "flush_count": 0,
            "file_opens": 0,
            "makedirs_calls": 0,
            "buffer_bytes": 0,
            "forced_flush_count": 0,
            "write_errors": 0,
            "lock_records_queued": 0,
            "lock_records_written": 0,
            "lock_flush_count": 0,
            "lock_file_opens": 0,
            "lock_makedirs_calls": 0,
            "lock_buffer_bytes": 0,
            "lock_forced_flush_count": 0,
            "lock_write_errors": 0,
        }
        self._dev_stage_transition_count_gen: int = 0
        self._dev_stage_eval_stage_counts_gen: Dict[str, int] = {}
        self._dev_d2_lock_event_count_gen: int = 0
        self.env_hotpath_expression_diagnostics = EnvHotpathExpressionDiagnostics.maybe_create(self)
        # -----------------------------
        # Device selection (policy + banks)
        # -----------------------------
        # Defaults preserve legacy behavior:
        # - device='auto' => CUDA if available else CPU
        # - bank_device='auto' => same as device
        self.requested_device = str(getattr(self.sim_cfg, "device", "auto") or "auto")
        self.requested_bank_device = str(getattr(self.sim_cfg, "bank_device", "auto") or "auto")
        self.device = _resolve_torch_device(self.requested_device)
        self.bank_device = (
            self.device
            if self.requested_bank_device.lower() in ("auto", "")
            else _resolve_torch_device(self.requested_bank_device)
        )
        sim_dev = self.requested_device.lower()
        validate_inheritance_execution_lane(
            self.sim_cfg,
            policy_device=self.device,
            bank_device=self.bank_device,
        )
        validate_live_modules_config(
            self.sim_cfg,
            device=(str(self.device) if sim_dev in {"auto", ""} else sim_dev),
            policy_stride=int(getattr(self.sim_cfg, "policy_tick_stride", 1) or 1),
        )
        validate_routing_config(
            self.sim_cfg,
            device=(str(self.device) if sim_dev in {"auto", ""} else sim_dev),
            policy_stride=int(getattr(self.sim_cfg, "policy_tick_stride", 1) or 1),
        )
        validate_output_routing_config(
            self.sim_cfg,
            device=(str(self.device) if sim_dev in {"auto", ""} else sim_dev),
            policy_stride=int(getattr(self.sim_cfg, "policy_tick_stride", 1) or 1),
        )
        validate_structural_development_config(
            self.sim_cfg,
            device=(str(self.device) if sim_dev in {"auto", ""} else sim_dev),
            policy_stride=int(getattr(self.sim_cfg, "policy_tick_stride", 1) or 1),
        )
        _validate_escap7_gated_recurrent_config(
            self.sim_cfg,
            policy_device=self.device,
            bank_device=self.bank_device,
        )
        _validate_escap8_sustainable_growth_config(
            self.sim_cfg,
            policy_device=self.device,
            bank_device=self.bank_device,
        )
        if structural_development_enabled(self.sim_cfg):
            runnercore_authority_flags = sorted(
                name
                for name, value in vars(self.sim_cfg).items()
                if name.startswith("runnercore")
                and "authoritative" in name
                and bool(value)
            )
            if runnercore_authority_flags:
                raise RuntimeError(
                    "ESCAP6_RUNNERCORE_AUTHORITY_UNSUPPORTED "
                    f"flags={runnercore_authority_flags!r}"
                )

        # Torch CPU threads (note: CPU% can exceed 100% on multi-core systems).
        threads = int(getattr(self.sim_cfg, "torch_threads", 0) or 0)
        if threads > 0:
            try:
                torch.set_num_threads(threads)
            except Exception:
                pass

        # Optional compilation flag, implemented in runner_policy.py.
        self._torch_compile_policy = bool(getattr(self.sim_cfg, "torch_compile_policy", False))
        self.memory_action_window_instrumentation = MemoryActionWindowInstrumentation.maybe_create(self)
        self.mem_max_live_level4_instrumentation = MemMaxLiveLevel4Instrumentation.maybe_create(self)
        self.prediction_demand_action_context_instrumentation = (
            PredictionDemandActionContextInstrumentation.maybe_create(self)
        )
        self.cxxcore23_live_direct_compact_diagnostics = LiveDirectCompactDiagnostics.maybe_create(self)
        self.branch_observability = BranchObservabilityWriter.maybe_create(self)
        try:
            self.scientific_chronicle = ScientificChronicleRecorder.maybe_create(self)
            self._scientific_chronicle_init_error = ""
        except ValueError:
            # Invalid or unsafe profiles fail during preflight, before mechanics.
            raise
        except Exception as exc:
            # Filesystem observability failures must not block the simulation.
            self.scientific_chronicle = None
            self._scientific_chronicle_init_error = f"{type(exc).__name__}: {exc}"
        try:
            self.gen1_f1_capture = Gen1F1Capture.maybe_create(self)
            self._gen1_f1_capture_init_error = ""
        except Exception as exc:
            # GEN1-F1 is observability-only and cannot block mechanics.
            self.gen1_f1_capture = None
            self._gen1_f1_capture_init_error = f"{type(exc).__name__}: {exc}"
        try:
            setattr(self.pop, "gen1_f1_capture", self.gen1_f1_capture)
        except Exception:
            pass


        self.obs_size = patch_world_total_observation_size(env_cfg)
        self.hidden_size = 64
        self.action_id_map = {
            "MOVE_N": 0,
            "MOVE_S": 1,
            "MOVE_E": 2,
            "MOVE_W": 3,
            "GATHER": 4,
            "REST": 5,
            "ATTACK": 6,
        }

        if getattr(self.sim_cfg, "behavior_log_enable", True):
            self.behavior_logger = BehaviorLogger(
                run_dir=self.run_dir,
                action_id_map=self.action_id_map,
                cohort_mode="full",
            )
        else:
            # Used by internal shadow-clone evaluation to avoid side-effecting the filesystem.
            self.behavior_logger = NoOpBehaviorLogger()
        self.mem_max_pre_cue_checkpoint_recorder = MemMaxPreCueCheckpointRecorder.maybe_create(self)

        # -----------------------------
        # GPU policy bank (per-slot parameter store)
        # -----------------------------
        self._bank_device = self.bank_device
        self._bank_dtype = torch.float32

        self._local_size = patch_world_local_observation_size(self.env_cfg)
        self._obs_size = patch_world_total_observation_size(self.env_cfg)
        self._hidden = 64
        self._out = 7

        self._W1 = None
        self._b1 = None
        self._W2 = None
        self._b2 = None
        self._W3 = None
        self._b3 = None

        # -----------------------------
        # Runner-level internal drives (per-slot arrays)
        # -----------------------------
        self._drive_hunger: Optional[np.ndarray] = None
        self._drive_pain: Optional[np.ndarray] = None
        self._drive_fatigue: Optional[np.ndarray] = None
        self._drive_fear: Optional[np.ndarray] = None
        self._drive_slot_token: Optional[np.ndarray] = None
        self._chemistry_state: Optional[np.ndarray] = None
        self._chemistry_slot_token: Optional[np.ndarray] = None
        self._maintenance_state: Optional[np.ndarray] = None
        self._maintenance_slot_token: Optional[np.ndarray] = None

        # -----------------------------
        # Runner-level identity drift state (per-slot arrays)
        # -----------------------------
        # We store probe baseline signature and drift EMA per slot for performance.
        # Slot reuse is handled via a parallel token array that must match agent.slot_token.
        self._idprobe_baseline_probs: Optional[np.ndarray] = None  # (max_slots, A)
        self._idprobe_baseline_token: Optional[np.ndarray] = None  # (max_slots,)
        self._idprobe_drift_ema: Optional[np.ndarray] = None       # (max_slots,)
        self._idprobe_last_tick: int = 0


        # Allocate drive arrays sized to the slot pool if available.
        self._ensure_drive_arrays()
        self._ensure_identity_arrays()

    def observe_substrate_capacity_pressure(self, sample: Any):
        manager = getattr(self, "substrate_capacity_manager", None)
        if manager is None:
            raise RuntimeError("substrate capacity manager is disabled")
        state = manager.observe(sample)
        self.sim_cfg.capacity_epoch_transition_state = state.value
        return state

    def request_substrate_capacity_migration(self, *, target_profile_id: str, generation: int) -> None:
        """Explicit integrity request; never an evolutionary or fitness decision."""

        manager = getattr(self, "substrate_capacity_manager", None)
        if manager is None:
            raise RuntimeError("substrate capacity manager is disabled")
        manager.authorize(target_profile_id=target_profile_id, generation=int(generation))
        self.sim_cfg.capacity_epoch_transition_state = manager.state.value

    def _dev_generation_index(self) -> int:
        try:
            value = int(getattr(self.pop, "_eiel_current_generation_index", -1))
            if value >= 0:
                return value
        except Exception:
            pass
        try:
            return int(getattr(self, "_eiel_fdp_cur_gen", -1))
        except Exception:
            return -1

    def _dev_tick_index(self) -> int:
        try:
            value = int(getattr(self, "_pred_tick", -1))
            if value >= 0:
                return value
        except Exception:
            pass
        try:
            return int(getattr(self, "_eiel_fdp_cur_tick", -1))
        except Exception:
            return -1

    def _dev_trace_path(self) -> str:
        return os.path.join(str(self.run_dir), "metrics", "developmental_stage_trace.jsonl")

    def _dev_lock_trace_path(self) -> str:
        return os.path.join(str(self.run_dir), "metrics", "developmental_lock_trace.jsonl")

    def _dev_buffered_trace_emit_enabled(self) -> bool:
        try:
            cfg = getattr(self, "sim_cfg", None)
            if not bool(getattr(cfg, "runnercore47_developmental_trace_buffered_emit_enable", False)):
                return False
            return int(getattr(cfg, "checkpoint_bundle_v2_period_gen", 0) or 0) == 0
        except Exception:
            return False

    def _dev_trace_write_enabled(self) -> bool:
        try:
            return bool(getattr(getattr(self, "sim_cfg", None), "developmental_trace_write_enable", True))
        except Exception:
            return True

    @staticmethod
    def _dev_trace_line(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"

    def _dev_trace_counter_add(self, key: str, value: int = 1) -> None:
        counters = getattr(self, "_dev_trace_emit_counters", None)
        if not isinstance(counters, dict):
            counters = {}
        counters[str(key)] = int(counters.get(str(key), 0) or 0) + int(value)
        self._dev_trace_emit_counters = counters

    def _dev_trace_ensure_parent_dir(self, out: str, *, lock: bool = False) -> None:
        d = os.path.dirname(str(out))
        if not d:
            return
        ready = getattr(self, "_dev_trace_emit_dirs_ready", None)
        if not isinstance(ready, set):
            ready = set()
        if d in ready:
            self._dev_trace_emit_dirs_ready = ready
            return
        os.makedirs(d, exist_ok=True)
        ready.add(d)
        self._dev_trace_emit_dirs_ready = ready
        self._dev_trace_counter_add("lock_makedirs_calls" if lock else "makedirs_calls")

    @staticmethod
    def _dev_previous_active_module_count(st: Any) -> int:
        raw = getattr(st, "_dev_active_module_count", None)
        if raw is None:
            return -1
        try:
            return int(raw)
        except Exception:
            return -1

    def _dev_flush_trace_emit_buffer(self, *, force: bool = True) -> None:
        buf = getattr(self, "_dev_trace_emit_buffer", None)
        if not isinstance(buf, list) or not buf:
            return
        try:
            out = self._dev_trace_path()
            self._dev_trace_ensure_parent_dir(out)
            with open(out, "a", encoding="utf-8") as f:
                self._dev_trace_counter_add("file_opens")
                f.writelines(str(line) for line in buf)
            self._dev_trace_counter_add("records_written", len(buf))
            self._dev_trace_counter_add("flush_count")
            if force:
                self._dev_trace_counter_add("forced_flush_count")
            buf.clear()
            self._dev_trace_emit_buffer = buf
        except Exception:
            self._dev_trace_counter_add("write_errors")
            raise

    def _dev_flush_lock_trace_emit_buffer(self, *, force: bool = True) -> None:
        buf = getattr(self, "_dev_lock_trace_emit_buffer", None)
        if not isinstance(buf, list) or not buf:
            return
        try:
            out = self._dev_lock_trace_path()
            self._dev_trace_ensure_parent_dir(out, lock=True)
            with open(out, "a", encoding="utf-8") as f:
                self._dev_trace_counter_add("lock_file_opens")
                f.writelines(str(line) for line in buf)
            self._dev_trace_counter_add("lock_records_written", len(buf))
            self._dev_trace_counter_add("lock_flush_count")
            if force:
                self._dev_trace_counter_add("lock_forced_flush_count")
            buf.clear()
            self._dev_lock_trace_emit_buffer = buf
        except Exception:
            self._dev_trace_counter_add("lock_write_errors")
            raise

    def _dev_emit_trace(self, payload: Dict[str, Any]) -> None:
        if not self._dev_trace_write_enabled():
            return
        line = self._dev_trace_line(payload)
        if self._dev_buffered_trace_emit_enabled():
            buf = getattr(self, "_dev_trace_emit_buffer", None)
            if not isinstance(buf, list):
                buf = []
            buf.append(line)
            self._dev_trace_emit_buffer = buf
            self._dev_trace_counter_add("records_queued")
            self._dev_trace_counter_add("buffer_bytes", len(line.encode("utf-8")))
            return
        try:
            out = self._dev_trace_path()
            self._dev_trace_ensure_parent_dir(out)
            with open(out, "a", encoding="utf-8") as f:
                self._dev_trace_counter_add("file_opens")
                f.write(line)
            self._dev_trace_counter_add("records_written")
        except Exception:
            self._dev_trace_counter_add("write_errors")
            raise

    def _dev_emit_lock_trace(self, payload: Dict[str, Any]) -> None:
        if not self._dev_trace_write_enabled():
            return
        line = self._dev_trace_line(payload)
        if self._dev_buffered_trace_emit_enabled():
            buf = getattr(self, "_dev_lock_trace_emit_buffer", None)
            if not isinstance(buf, list):
                buf = []
            buf.append(line)
            self._dev_lock_trace_emit_buffer = buf
            self._dev_trace_counter_add("lock_records_queued")
            self._dev_trace_counter_add("lock_buffer_bytes", len(line.encode("utf-8")))
            return
        try:
            out = self._dev_lock_trace_path()
            self._dev_trace_ensure_parent_dir(out, lock=True)
            with open(out, "a", encoding="utf-8") as f:
                self._dev_trace_counter_add("lock_file_opens")
                f.write(line)
            self._dev_trace_counter_add("lock_records_written")
        except Exception:
            self._dev_trace_counter_add("lock_write_errors")
            raise

    def _dev_flush_pending_traces(self, *, force: bool = False, upto_key: tuple[int, int] | None = None) -> None:
        pending = getattr(self, "_dev_trace_pending", None)
        if not isinstance(pending, dict) or not pending:
            return
        keys = sorted((k for k in pending.keys() if isinstance(k, tuple) and len(k) == 2), key=lambda x: (int(x[0]), int(x[1])))
        for key in keys:
            if not force:
                if upto_key is not None:
                    if not (int(key[0]) < int(upto_key[0]) or (int(key[0]) == int(upto_key[0]) and int(key[1]) < int(upto_key[1]))):
                        continue
                else:
                    continue
            bucket = pending.get(key, {})
            if not isinstance(bucket, dict):
                pending.pop(key, None)
                continue
            for aid in sorted((int(x) for x in bucket.keys())):
                payload = bucket.get(int(aid), None)
                if isinstance(payload, dict):
                    self._dev_emit_trace(payload)
            pending.pop(key, None)
        self._dev_trace_pending = pending

    def _dev_queue_trace(self, payload: Dict[str, Any]) -> None:
        if not self._dev_trace_write_enabled():
            return
        try:
            g = int(payload.get("generation", -1))
            t = int(payload.get("tick", -1))
            aid = int(payload.get("agent_id", 0))
        except Exception:
            self._dev_emit_trace(payload)
            return
        pending = getattr(self, "_dev_trace_pending", None)
        if not isinstance(pending, dict):
            pending = {}
        key = (int(g), int(t))
        bucket = pending.get(key)
        if not isinstance(bucket, dict):
            bucket = {}
            pending[key] = bucket
        if int(aid) not in bucket:
            bucket[int(aid)] = payload
        self._dev_trace_pending = pending
        self._dev_flush_pending_traces(force=False, upto_key=key)

    def _dev_flush_pending_lock_traces(self, *, force: bool = False, upto_key: tuple[int, int] | None = None) -> None:
        pending = getattr(self, "_dev_lock_trace_pending", None)
        if not isinstance(pending, dict) or not pending:
            return
        keys = sorted((k for k in pending.keys() if isinstance(k, tuple) and len(k) == 2), key=lambda x: (int(x[0]), int(x[1])))
        for key in keys:
            if not force:
                if upto_key is not None:
                    if not (int(key[0]) < int(upto_key[0]) or (int(key[0]) == int(upto_key[0]) and int(key[1]) < int(upto_key[1]))):
                        continue
                else:
                    continue
            bucket = pending.get(key, {})
            if not isinstance(bucket, dict):
                pending.pop(key, None)
                continue
            for aid in sorted((int(x) for x in bucket.keys())):
                payload = bucket.get(int(aid), None)
                if isinstance(payload, dict):
                    self._dev_emit_lock_trace(payload)
            pending.pop(key, None)
        self._dev_lock_trace_pending = pending

    def _dev_queue_lock_trace(self, payload: Dict[str, Any]) -> None:
        if not self._dev_trace_write_enabled():
            return
        try:
            g = int(payload.get("generation", -1))
            t = int(payload.get("tick", -1))
            aid = int(payload.get("agent_id", 0))
        except Exception:
            self._dev_emit_lock_trace(payload)
            return
        pending = getattr(self, "_dev_lock_trace_pending", None)
        if not isinstance(pending, dict):
            pending = {}
        key = (int(g), int(t))
        bucket = pending.get(key)
        if not isinstance(bucket, dict):
            bucket = {}
            pending[key] = bucket
        if int(aid) not in bucket:
            bucket[int(aid)] = payload
        self._dev_lock_trace_pending = pending
        self._dev_flush_pending_lock_traces(force=False, upto_key=key)

    def _dev_effective_scalar_cache_enabled(self) -> bool:
        try:
            return bool(getattr(self.sim_cfg, "cxxrewrite_resident_dev_scalar_cache_enable", False))
        except Exception:
            return False

    def _dev_stage_cache_enabled(self) -> bool:
        try:
            return bool(getattr(self.sim_cfg, "cxxrewrite_resident_dev_stage_cache_enable", False))
        except Exception:
            return False

    def _dev_stage_bounds_cache_enabled(self) -> bool:
        try:
            return bool(getattr(self.sim_cfg, "cxxrewrite_resident_dev_stage_bounds_cache_enable", False))
        except Exception:
            return False

    def _dev_expression_cache_key_for_agent(self, agent: Any) -> tuple[int, int, int, int]:
        aid = int(getattr(agent, "agent_id", 0) or 0)
        st = getattr(agent, "stats", None)
        age = int(getattr(st, "survival_steps", 0) or 0) if st is not None else 0
        return (int(aid), int(self._dev_generation_index()), int(self._dev_tick_index()), int(age))

    @staticmethod
    def _dev_effective_scalars_from_expression(agent: Any, expr: Dict[str, Any] | None) -> tuple[float, ...]:
        # Tuple order: aggression_bias, action_temperature, drive_logit_gain,
        # drive_inertia, drive_noise, pred_lr, reproduction_threshold.
        g = getattr(agent, "genome", None)
        eff = expr.get("effective", {}) if isinstance(expr, dict) else {}
        if not isinstance(eff, dict):
            eff = {}
        if g is None:
            return (0.0, 1.0, 1.0, 0.85, 0.02, 0.08, 50.0)
        return (
            float(eff.get("aggression_bias", getattr(g, "aggression_bias", 0.0))),
            float(eff.get("action_temperature", getattr(g, "action_temperature", 1.0))),
            float(eff.get("drive_logit_gain", getattr(g, "drive_logit_gain", 1.0))),
            float(eff.get("drive_inertia", getattr(g, "drive_inertia", 0.85))),
            float(eff.get("drive_noise", getattr(g, "drive_noise", 0.02))),
            float(eff.get("pred_lr", getattr(g, "pred_lr", 0.08))),
            float(eff.get("reproduction_threshold", getattr(g, "reproduction_threshold", 50.0))),
        )

    def resolve_developmental_effective_scalars_for_agent(self, agent: Any) -> tuple[float, ...] | None:
        if not self._dev_effective_scalar_cache_enabled():
            return None
        key = self._dev_expression_cache_key_for_agent(agent)
        cache = getattr(self, "_dev_effective_scalar_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._dev_effective_scalar_cache = cache
        cached = cache.get(key)
        if isinstance(cached, tuple) and len(cached) >= 7:
            return cached
        expr = self.resolve_developmental_expression_for_agent(agent)
        scalars = self._dev_effective_scalars_from_expression(agent, expr if isinstance(expr, dict) else None)
        cache[key] = scalars
        return scalars

    @staticmethod
    def _dev_stage_cache_guard(genome: Any, stage_id: str) -> tuple[str, int, int, int]:
        return (
            str(stage_id),
            int(id(genome)),
            int(id(getattr(genome, "developmental_program", None))),
            int(id(getattr(genome, "modules", None))),
        )

    @staticmethod
    def _dev_stage_bounds_for_age(genome: Any, *, stage_id: str, age_steps: int) -> tuple[int, int] | None:
        try:
            program = canonicalize_developmental_program(getattr(genome, "developmental_program", None))
            age = max(0, int(age_steps))
            target_stage = str(stage_id or "default")
            for rec in program.get("stages", []) or []:
                start = int(rec.get("start_step", 0) or 0)
                end = int(rec.get("end_step", -1) or -1)
                if age < start:
                    continue
                if end >= 0 and age > end:
                    continue
                if str(rec.get("stage_id", "default") or "default") != target_stage:
                    continue
                return (int(start), int(end))
        except Exception:
            return None
        return None

    def _dev_stage_cache_apply_expr(self, agent: Any, expr: Dict[str, Any], *, stage_id: str) -> None:
        st = getattr(agent, "stats", None)
        if st is None:
            return
        try:
            setattr(st, "_dev_stage_id", str(expr.get("stage_id", stage_id) or stage_id))
            setattr(st, "_dev_active_module_count", int(expr.get("active_module_count", 0) or 0))
            setattr(st, "_dev_last_trace_gen", int(self._dev_generation_index()))
            setattr(st, "_dev_last_trace_tick", int(self._dev_tick_index()))
        except Exception:
            pass

    def _dev_stage_cache_lookup_by_bounds(self, agent: Any, genome: Any, *, age_steps: int) -> Dict[str, Any] | None:
        st = getattr(agent, "stats", None)
        if st is None or not self._dev_stage_bounds_cache_enabled():
            return None
        cached = getattr(st, "_dev_stage_expr_cache", None)
        if not isinstance(cached, dict):
            return None
        expr = cached.get("expression")
        if not isinstance(expr, dict):
            return None
        stage_id = str(cached.get("stage_id", expr.get("stage_id", "default")) or "default")
        if cached.get("guard") != self._dev_stage_cache_guard(genome, stage_id):
            return None
        try:
            age = max(0, int(age_steps))
            start = int(cached.get("stage_start_step", 0) or 0)
            end = int(cached.get("stage_end_step", -1) or -1)
        except Exception:
            return None
        if age < start:
            return None
        if end >= 0 and age > end:
            return None
        self._dev_stage_cache_apply_expr(agent, expr, stage_id=stage_id)
        return expr

    def _dev_stage_cache_lookup(self, agent: Any, genome: Any, *, age_steps: int) -> Dict[str, Any] | None:
        st = getattr(agent, "stats", None)
        if st is None or not self._dev_stage_cache_enabled():
            return None
        bounds_cached = self._dev_stage_cache_lookup_by_bounds(agent, genome, age_steps=int(age_steps))
        if isinstance(bounds_cached, dict):
            return bounds_cached
        try:
            stage_id = str(genome.developmental_stage_id(age_steps=int(age_steps)))
        except Exception:
            return None
        cached = getattr(st, "_dev_stage_expr_cache", None)
        if not isinstance(cached, dict):
            return None
        guard = self._dev_stage_cache_guard(genome, stage_id)
        if cached.get("guard") != guard:
            return None
        expr = cached.get("expression")
        if not isinstance(expr, dict):
            return None
        self._dev_stage_cache_apply_expr(agent, expr, stage_id=stage_id)
        return expr

    def _dev_stage_cache_store(self, agent: Any, genome: Any, expr: Dict[str, Any], *, age_steps: int | None = None) -> None:
        st = getattr(agent, "stats", None)
        if st is None or not self._dev_stage_cache_enabled() or not isinstance(expr, dict):
            return
        try:
            stage_id = str(expr.get("stage_id", "default") or "default")
            record = {
                "guard": self._dev_stage_cache_guard(genome, stage_id),
                "stage_id": stage_id,
                "expression": copy.deepcopy(expr),
            }
            if self._dev_stage_bounds_cache_enabled():
                age = int(age_steps) if age_steps is not None else int(getattr(st, "survival_steps", 0) or 0)
                bounds = self._dev_stage_bounds_for_age(genome, stage_id=stage_id, age_steps=int(age))
                if bounds is not None:
                    record["stage_start_step"] = int(bounds[0])
                    record["stage_end_step"] = int(bounds[1])
            setattr(
                st,
                "_dev_stage_expr_cache",
                record,
            )
        except Exception:
            pass

    def resolve_developmental_expression_for_agent(self, agent: Any) -> Dict[str, Any]:
        key = self._dev_expression_cache_key_for_agent(agent)
        aid, gidx, tidx, age = key
        st = getattr(agent, "stats", None)
        cached = self._dev_expr_cache.get(key)
        diag = getattr(self, "env_hotpath_expression_diagnostics", None)
        if diag is not None:
            try:
                diag.record_dev_expression_request(
                    agent_id=int(aid),
                    generation=int(gidx),
                    tick=int(tidx),
                    age_steps=int(age),
                    cache_hit=isinstance(cached, dict),
                    has_genome=getattr(agent, "genome", None) is not None,
                )
            except Exception:
                pass
        if isinstance(cached, dict):
            if self._dev_effective_scalar_cache_enabled():
                scalar_cache = getattr(self, "_dev_effective_scalar_cache", None)
                if not isinstance(scalar_cache, dict):
                    scalar_cache = {}
                    self._dev_effective_scalar_cache = scalar_cache
                if key not in scalar_cache:
                    scalar_cache[key] = self._dev_effective_scalars_from_expression(agent, cached)
            return cached

        g = getattr(agent, "genome", None)
        if g is None or not hasattr(g, "resolve_developmental_expression"):
            out = {
                "stage_id": "default",
                "active_module_count": 0,
                "module_count": 0,
                "active_by_module_id": {},
                "inactive_reason_by_module_id": {},
                "module_structure_signature": "",
                "effective": {},
                "structural_change_ignored": False,
                "structural_change_reason": "",
                "lock_state_by_module_id": {},
                "lock_events": [],
            }
            self._dev_expr_cache[key] = out
            if self._dev_effective_scalar_cache_enabled():
                self._dev_effective_scalar_cache[key] = self._dev_effective_scalars_from_expression(agent, out)
            return out

        stage_cached = self._dev_stage_cache_lookup(agent, g, age_steps=int(age))
        if isinstance(stage_cached, dict):
            self._dev_expr_cache[key] = stage_cached
            if self._dev_effective_scalar_cache_enabled():
                self._dev_effective_scalar_cache[key] = self._dev_effective_scalars_from_expression(agent, stage_cached)
            return stage_cached

        expr = g.resolve_developmental_expression(age_steps=int(age))
        expr["structural_change_ignored"] = False
        expr["structural_change_reason"] = ""
        expr["lock_state_by_module_id"] = {}
        expr["lock_events"] = []
        expr["lock_event_count"] = 0

        strict_mode = str(os.environ.get("EIEL_STRICT_DETERMINISM", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        prev_stage = ""
        prev_lock_state = {}
        if st is not None:
            prev_stage = str(getattr(st, "_dev_stage_id", "") or "")
            lock_state_raw = getattr(st, "_dev_d2_lock_state_by_module_id", {})
            if isinstance(lock_state_raw, dict):
                prev_lock_state = dict(lock_state_raw)
        try:
            lock_result = apply_irreversible_stage_locks(
                program=getattr(g, "developmental_program", None),
                stage_id=str(expr.get("stage_id", "default") or "default"),
                prev_stage_id=prev_stage,
                active_by_module_id=dict(expr.get("active_by_module_id", {}) or {}),
                inactive_reason_by_module_id=dict(expr.get("inactive_reason_by_module_id", {}) or {}),
                previous_lock_state=prev_lock_state,
                strict=bool(strict_mode),
            )
            expr["active_by_module_id"] = dict(lock_result.get("active_by_module_id", {}) or {})
            expr["inactive_reason_by_module_id"] = dict(lock_result.get("inactive_reason_by_module_id", {}) or {})
            expr["lock_state_by_module_id"] = dict(lock_result.get("lock_state_by_module_id", {}) or {})
            expr["lock_events"] = list(lock_result.get("events", []) or [])
            expr["lock_event_count"] = int(len(expr["lock_events"]))
            expr["active_module_count"] = int(sum(1 for _v in expr["active_by_module_id"].values() if bool(_v)))
            expr["module_count"] = int(len(expr["active_by_module_id"]))
        except RuntimeError:
            raise
        except Exception:
            pass

        if structural_live_semantics_enabled(getattr(self, "sim_cfg", None)):
            expr, _live_summary = apply_live_structural_semantics(
                genome=g,
                expression=expr,
                sim_cfg=getattr(self, "sim_cfg", None),
            )

        baseline_sig = ""
        if st is not None:
            baseline_sig = str(getattr(st, "_dev_module_structure_signature", "") or "")
        current_sig = str(expr.get("module_structure_signature", "") or "")

        if st is not None and not baseline_sig:
            try:
                setattr(st, "_dev_module_structure_signature", current_sig)
            except Exception:
                pass
        elif st is not None and baseline_sig and baseline_sig != current_sig:
            # D1 rule: structure must not change mid-life. Keep the prior resolved
            # expression snapshot so observability and module activity maps remain stable.
            expr["structural_change_ignored"] = True
            expr["structural_change_reason"] = "runtime_structure_change_ignored"
            prev_expr = getattr(st, "_dev_last_expr_snapshot", None)
            if isinstance(prev_expr, dict):
                for k in (
                    "stage_id",
                    "active_module_count",
                    "module_count",
                    "active_by_module_id",
                    "inactive_reason_by_module_id",
                    "effective",
                    "lock_state_by_module_id",
                    "structural_live_semantics",
                ):
                    if k in prev_expr:
                        expr[k] = prev_expr[k]
                expr["lock_events"] = []
                expr["lock_event_count"] = 0
            else:
                prev_eff = getattr(st, "_dev_last_effective", None)
                if isinstance(prev_eff, dict):
                    expr["effective"] = dict(prev_eff)
                prev_locks = getattr(st, "_dev_d2_lock_state_by_module_id", None)
                if isinstance(prev_locks, dict):
                    expr["lock_state_by_module_id"] = dict(prev_locks)
                expr["lock_events"] = []
                expr["lock_event_count"] = 0

        if st is not None:
            prev_stage = str(getattr(st, "_dev_stage_id", "") or "")
            prev_active = self._dev_previous_active_module_count(st)
            stage_id = str(expr.get("stage_id", "default") or "default")
            active_count = int(expr.get("active_module_count", 0) or 0)
            lock_state = dict(expr.get("lock_state_by_module_id", {}) or {})
            lock_state_json = json.dumps(dict(sorted(lock_state.items(), key=lambda kv: str(kv[0]))), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            lock_state_sha = hashlib.sha256(lock_state_json.encode("utf-8")).hexdigest()
            if (stage_id != prev_stage) or (active_count != prev_active) or bool(expr.get("structural_change_ignored", False)):
                self._dev_queue_trace(
                    {
                        "schema_version": "developmental_stage_trace_v1",
                        "generation": int(gidx),
                        "tick": int(tidx),
                        "agent_id": int(aid),
                        "age_steps": int(age),
                        "stage_id": stage_id,
                        "active_module_count": int(active_count),
                        "module_count": int(expr.get("module_count", 0) or 0),
                        "locked_module_count": int(len(lock_state)),
                        "lock_state_sha256": str(lock_state_sha),
                        "structural_change_ignored": bool(expr.get("structural_change_ignored", False)),
                        "structural_change_reason": str(expr.get("structural_change_reason", "")),
                        "inactive_reason_counts": self._dev_reason_counts(expr),
                    }
                )
                self._dev_stage_transition_count_gen = int(getattr(self, "_dev_stage_transition_count_gen", 0) or 0) + 1
                counts = getattr(self, "_dev_stage_eval_stage_counts_gen", None)
                if not isinstance(counts, dict):
                    counts = {}
                counts[str(stage_id)] = int(counts.get(str(stage_id), 0) or 0) + 1
                self._dev_stage_eval_stage_counts_gen = counts
            lock_events = list(expr.get("lock_events", []) or [])
            if lock_events:
                self._dev_queue_lock_trace(
                    {
                        "schema_version": "developmental_lock_trace_v1",
                        "generation": int(gidx),
                        "tick": int(tidx),
                        "agent_id": int(aid),
                        "age_steps": int(age),
                        "stage_id": stage_id,
                        "lock_state_sha256": str(lock_state_sha),
                        "lock_state_by_module_id": dict(sorted(lock_state.items(), key=lambda kv: str(kv[0]))),
                        "events": lock_events,
                    }
                )
                self._dev_d2_lock_event_count_gen = int(getattr(self, "_dev_d2_lock_event_count_gen", 0) or 0) + int(len(lock_events))
            setattr(st, "_dev_stage_id", stage_id)
            setattr(st, "_dev_active_module_count", int(active_count))
            setattr(st, "_dev_last_effective", dict(expr.get("effective", {}) or {}))
            setattr(st, "_dev_d2_lock_state_by_module_id", dict(sorted(lock_state.items(), key=lambda kv: str(kv[0]))))
            setattr(st, "_dev_d2_lock_state_sha256", str(lock_state_sha))
            setattr(
                st,
                "_dev_last_expr_snapshot",
                {
                    "stage_id": str(expr.get("stage_id", "default") or "default"),
                    "active_module_count": int(expr.get("active_module_count", 0) or 0),
                    "module_count": int(expr.get("module_count", 0) or 0),
                    "active_by_module_id": dict(expr.get("active_by_module_id", {}) or {}),
                    "inactive_reason_by_module_id": dict(expr.get("inactive_reason_by_module_id", {}) or {}),
                    "effective": dict(expr.get("effective", {}) or {}),
                    "lock_state_by_module_id": dict(sorted(lock_state.items(), key=lambda kv: str(kv[0]))),
                    "structural_live_semantics": dict(expr.get("structural_live_semantics", {}) or {}),
                },
            )
            setattr(st, "_dev_last_trace_gen", int(gidx))
            setattr(st, "_dev_last_trace_tick", int(tidx))

        if diag is not None and (
            bool(getattr(diag, "developmental_cache_equality_enabled", False))
            or bool(getattr(diag, "developmental_effective_projection_enabled", False))
        ):
            try:
                lock_state = dict(expr.get("lock_state_by_module_id", {}) or {})
                lock_state_json = json.dumps(
                    dict(sorted(lock_state.items(), key=lambda kv: str(kv[0]))),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                lock_state_sha = hashlib.sha256(lock_state_json.encode("utf-8")).hexdigest()
                program_json = json.dumps(
                    getattr(g, "developmental_program", None),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    default=str,
                )
                program_sha = hashlib.sha256(program_json.encode("utf-8")).hexdigest()
                setattr(
                    agent,
                    "_eiel_dev_expression_cache_candidate",
                    {
                        "schema_version": "envhot8_dev_expression_candidate_v1",
                        "agent_id": int(aid),
                        "generation": int(gidx),
                        "tick": int(tidx),
                        "age_steps": int(age),
                        "genome_obj_id": int(id(g)),
                        "program_sha256": str(program_sha),
                        "lock_state_sha256": str(lock_state_sha),
                        "expression": copy.deepcopy(expr),
                    },
                )
                diag.record_developmental_cache_candidate(
                    agent_id=int(aid),
                    generation=int(gidx),
                    tick=int(tidx),
                    age_steps=int(age),
                    lock_state_sha256=str(lock_state_sha),
                )
            except Exception:
                pass

        self._dev_expr_cache[key] = expr
        self._dev_stage_cache_store(agent, g, expr, age_steps=int(age))
        if self._dev_effective_scalar_cache_enabled():
            self._dev_effective_scalar_cache[key] = self._dev_effective_scalars_from_expression(agent, expr)
        return expr

    @staticmethod
    def _dev_reason_counts(expr: Dict[str, Any]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        reasons = expr.get("inactive_reason_by_module_id", {}) if isinstance(expr, dict) else {}
        if not isinstance(reasons, dict):
            return out
        for _mid, reason in reasons.items():
            rs = str(reason or "")
            out[rs] = int(out.get(rs, 0)) + 1
        return dict(sorted(out.items(), key=lambda kv: kv[0]))

    def _enforce_b2_campaign_freeze(self) -> None:
        """Normalize/validate B2 governance knobs against campaign-frozen constants.

        These knobs are exposed in SimConfig for audit visibility and run-config capture.
        Runtime B2 canonicalization remains fixed to governance constants.
        """
        strict = bool(_b2_strict_enabled())
        cfg = self.sim_cfg
        cxx26_scope_lock = (
            bool(getattr(cfg, "native_runtime_cxx26_scope_lock_enable", False))
            and str(getattr(cfg, "native_runtime_fixed_policy_family", "") or "").strip().lower() == "mlp"
        )

        if hasattr(cfg, "genome_b2_interfaces_enable"):
            raw = bool(getattr(cfg, "genome_b2_interfaces_enable", True))
            if raw is not True and strict:
                raise RuntimeError(
                    "[DETERMINISM_VIOLATION]\n"
                    "code=DET_GENOME_B2_CAP_FREEZE_MISMATCH\n"
                    "context=field=genome_b2_interfaces_enable;expected=True;observed=False\n"
                )
            setattr(cfg, "genome_b2_interfaces_enable", True)

        # Legacy/non-strict loading retains compatibility clipping.  Strict
        # scientific execution upgrades the active boundary to reject so newly
        # produced hereditary overflow cannot become silently valid science.
        policy_expected = "reject" if strict else "clip"
        if hasattr(cfg, "genome_b2_invalid_policy"):
            policy = str(getattr(cfg, "genome_b2_invalid_policy", policy_expected) or policy_expected).strip().lower()
            if policy not in {"clip", "reject"} and strict:
                raise RuntimeError(
                    "[DETERMINISM_VIOLATION]\n"
                    "code=DET_GENOME_B2_CAP_FREEZE_MISMATCH\n"
                    f"context=field=genome_b2_invalid_policy;expected=clip_or_reject;observed={policy}\n"
                )
            setattr(cfg, "genome_b2_invalid_policy", policy_expected)

        cap_fields = {
            "max_modules_per_agent": "genome_b2_max_modules_per_agent",
            "max_connections_per_agent": "genome_b2_max_connections_per_agent",
            "max_module_fanin": "genome_b2_max_module_fanin",
            "max_module_fanout": "genome_b2_max_module_fanout",
        }
        requested_caps = {
            cap_name: getattr(cfg, field, B2_FROZEN_CAPS[cap_name])
            for cap_name, field in cap_fields.items()
        }
        try:
            resolved_caps = resolve_b2_caps(requested_caps)
        except ValueError as exc:
            raise RuntimeError(
                "[DETERMINISM_VIOLATION]\n"
                "code=DET_GENOME_B2_CAP_FREEZE_MISMATCH\n"
                f"context=requested_caps={requested_caps!r};reason={exc}\n"
            ) from exc
        for cap_name, field in cap_fields.items():
            if hasattr(cfg, field):
                setattr(cfg, field, int(resolved_caps[cap_name]))

        b3_bool_expected = True
        if hasattr(cfg, "genome_b3_locality_enable"):
            raw = bool(getattr(cfg, "genome_b3_locality_enable", True))
            if raw is not b3_bool_expected and strict:
                raise RuntimeError(
                    "[DETERMINISM_VIOLATION]\n"
                    "code=DET_GENOME_B3_FREEZE_MISMATCH\n"
                    "context=field=genome_b3_locality_enable;expected=True;observed=False\n"
                )
            setattr(cfg, "genome_b3_locality_enable", True)

        b3_float_fields = {
            "genome_b3_max_connection_distance": float(B3_FROZEN_LOCALITY["max_connection_distance"]),
            "genome_b3_distance_cost_scale": float(B3_FROZEN_LOCALITY["distance_cost_scale"]),
            "genome_b3_distance_cost_max": float(B3_FROZEN_LOCALITY["distance_cost_max"]),
        }
        for field, expected in b3_float_fields.items():
            if not hasattr(cfg, field):
                continue
            raw = getattr(cfg, field, expected)
            try:
                observed = float(raw)
            except Exception:
                observed = float(expected)
            if abs(observed - float(expected)) > 1e-12 and strict:
                raise RuntimeError(
                    "[DETERMINISM_VIOLATION]\n"
                    "code=DET_GENOME_B3_FREEZE_MISMATCH\n"
                    f"context=field={field};expected={expected};observed={observed}\n"
                )
            setattr(cfg, field, float(expected))

        c1_fields = {
            "genome_c1_enable": not cxx26_scope_lock,
            "genome_c1_catalog_version": str(C1_OPERATOR_CATALOG_VERSION),
            "genome_c1_max_ops_per_offspring": int(C1_FROZEN_MAX_OPS_PER_OFFSPRING),
            "genome_c1_invalid_operator_policy": str(C1_FROZEN_INVALID_OPERATOR_POLICY),
            "structural_identity_schema_version": str(STRUCTURAL_IDENTITY_SCHEMA_VERSION),
            "structural_ancestry_schema_version": str(STRUCTURAL_ANCESTRY_SCHEMA_VERSION),
            "topology_position_schema_version": str(TOPOLOGY_POSITION_SCHEMA_VERSION),
        }
        for field, expected in c1_fields.items():
            if not hasattr(cfg, field):
                continue
            raw = getattr(cfg, field, expected)
            observed: Any = raw
            if isinstance(expected, bool):
                observed = bool(raw)
                mismatch = observed is not bool(expected)
            elif isinstance(expected, int):
                try:
                    observed = int(raw)
                except Exception:
                    observed = int(expected)
                mismatch = int(observed) != int(expected)
            else:
                observed = str(raw if raw is not None else expected).strip()
                mismatch = str(observed) != str(expected)
            if mismatch and strict:
                raise RuntimeError(
                    "[DETERMINISM_VIOLATION]\n"
                    "code=DET_GENOME_C1_FREEZE_MISMATCH\n"
                    f"context=field={field};expected={expected};observed={observed}\n"
                )
            setattr(cfg, field, expected)

        if hasattr(cfg, "structural_bootstrap_enable"):
            bootstrap_enabled = bool(getattr(cfg, "structural_bootstrap_enable", False))
            if bootstrap_enabled and cxx26_scope_lock:
                raise RuntimeError(
                    "[DETERMINISM_VIOLATION]\n"
                    "code=DET_STRUCTURAL_BOOTSTRAP_UNSUPPORTED_EXECUTION_LANE\n"
                    "context=native_runtime_cxx26_scope_lock_enable=True\n"
                )
            setattr(cfg, "structural_bootstrap_enable", bootstrap_enabled)

        c2_fields = {
            "genome_c2_enable": not cxx26_scope_lock,
            "genome_c2_recombination_version": str(C2_RECOMBINATION_VERSION),
            "genome_c2_alignment_key": str(C2_FROZEN_ALIGNMENT_KEY),
            "genome_c2_unmatched_policy": str(C2_FROZEN_UNMATCHED_POLICY),
            "genome_c2_legacy_unresolved_policy": str(C2_LEGACY_UNRESOLVED_POLICY),
        }
        for field, expected in c2_fields.items():
            if not hasattr(cfg, field):
                continue
            raw = getattr(cfg, field, expected)
            observed: Any = raw
            if isinstance(expected, bool):
                observed = bool(raw)
                mismatch = observed is not bool(expected)
            else:
                observed = str(raw if raw is not None else expected).strip()
                mismatch = str(observed) != str(expected)
            if mismatch and strict:
                raise RuntimeError(
                    "[DETERMINISM_VIOLATION]\n"
                    "code=DET_GENOME_C2_FREEZE_MISMATCH\n"
                    f"context=field={field};expected={expected};observed={observed}\n"
                )
            setattr(cfg, field, expected)

    def _ensure_drive_arrays(self, required_slots: int = 0) -> None:
        """Ensure runner-level drive arrays exist and match slot capacity.

        This is safe to call frequently; it is a no-op once arrays are large enough.
        """
        sp = getattr(self.pop, "slot_pool", None)
        if sp is None:
            # Population stores its SlotPool as `_slots` in evolution/population_pkg/core.py
            sp = getattr(self.pop, "_slots", None)

        max_slots = int(getattr(sp, "max_slots", 0) or 0)

        # Fallbacks if SlotPool is not reachable for some reason
        if max_slots <= 0:
            max_slots = int(getattr(self.pop, "max_slots", 0) or 0)

        if max_slots <= 0:
            # Fallback: size from population config if slot pool is unavailable.
            base = int(getattr(self.pop_cfg, "population_size", 256) or 256)
            factor = int(getattr(self.pop_cfg, "slot_factor", 2) or 2)
            if factor < 2:
                factor = 2
            max_slots = max(1, base * factor)

        # Allow callers to demand a minimum size (e.g. max(slot)+1)
        if required_slots and required_slots > max_slots:
            max_slots = int(required_slots)

        # Already sufficient
        if self._drive_hunger is not None and int(self._drive_hunger.shape[0]) >= max_slots:
            if self._drive_slot_token is not None and int(self._drive_slot_token.shape[0]) >= max_slots:
                return

        def _grow(old: Optional[np.ndarray], fill: float = 0.0) -> np.ndarray:
            if old is None:
                return np.full((max_slots,), fill, dtype=np.float32)
            new = np.full((max_slots,), fill, dtype=np.float32)
            n = min(int(old.shape[0]), max_slots)
            new[:n] = old[:n].astype(np.float32, copy=False)
            return new

        self._drive_hunger = _grow(self._drive_hunger, fill=0.0)
        self._drive_pain = _grow(self._drive_pain, fill=0.0)
        self._drive_fatigue = _grow(self._drive_fatigue, fill=0.0)
        self._drive_fear = _grow(self._drive_fear, fill=0.0)

        if self._drive_slot_token is None or int(self._drive_slot_token.shape[0]) < max_slots:
            # IMPORTANT (Phase 3 determinism): initialize tokens to -1 so that
            # the first sync for any agent always triggers an explicit
            # initialization from the agent mirror (or zeros). If we default
            # to 0, agents with slot_token==0 will appear "already initialized"
            # and runner drives will remain stale zeros after resume.
            tok = np.full((max_slots,), -1, dtype=np.int32)
            if self._drive_slot_token is not None:
                n = min(int(self._drive_slot_token.shape[0]), max_slots)
                tok[:n] = self._drive_slot_token[:n].astype(np.int32, copy=False)
            self._drive_slot_token = tok

    def _ensure_chemistry_arrays(self, required_slots: int = 0) -> None:
        """Ensure runner-level chemistry arrays exist and match slot capacity."""
        sp = getattr(self.pop, "slot_pool", None)
        if sp is None:
            sp = getattr(self.pop, "_slots", None)

        max_slots = int(getattr(sp, "max_slots", 0) or 0)
        if max_slots <= 0:
            max_slots = int(getattr(self.pop, "max_slots", 0) or 0)
        if max_slots <= 0:
            base = int(getattr(self.pop_cfg, "population_size", 256) or 256)
            factor = int(getattr(self.pop_cfg, "slot_factor", 2) or 2)
            if factor < 2:
                factor = 2
            max_slots = max(1, base * factor)
        if required_slots and required_slots > max_slots:
            max_slots = int(required_slots)

        if self._chemistry_state is not None and int(self._chemistry_state.shape[0]) >= max_slots:
            if self._chemistry_slot_token is not None and int(self._chemistry_slot_token.shape[0]) >= max_slots:
                return

        if self._chemistry_state is None:
            chem = np.zeros((max_slots, CHEMISTRY_SIZE), dtype=np.float32)
        else:
            old_n = int(self._chemistry_state.shape[0])
            chem = np.zeros((max_slots, CHEMISTRY_SIZE), dtype=np.float32)
            chem[: min(old_n, max_slots), :] = self._chemistry_state[: min(old_n, max_slots), :].astype(
                np.float32,
                copy=False,
            )
        self._chemistry_state = chem

        if self._chemistry_slot_token is None or int(self._chemistry_slot_token.shape[0]) < max_slots:
            tok = np.full((max_slots,), -1, dtype=np.int32)
            if self._chemistry_slot_token is not None:
                n = min(int(self._chemistry_slot_token.shape[0]), max_slots)
                tok[:n] = self._chemistry_slot_token[:n].astype(np.int32, copy=False)
            self._chemistry_slot_token = tok

    def _sync_chemistry_slot(self, a, *, force_from_agent: bool = False) -> bool:
        """Initialize or refresh the resident chemistry row for one agent slot."""
        self._ensure_chemistry_arrays()
        slot = int(getattr(a, "slot", -1))
        if slot < 0:
            return False
        if self._chemistry_state is None or self._chemistry_slot_token is None:
            return False
        if slot >= int(self._chemistry_slot_token.shape[0]):
            self._ensure_chemistry_arrays(required_slots=slot + 1)
            if self._chemistry_state is None or self._chemistry_slot_token is None:
                return False
            if slot >= int(self._chemistry_slot_token.shape[0]):
                return False

        token = int(getattr(a, "slot_token", 0))
        if not force_from_agent and int(self._chemistry_slot_token[slot]) == token:
            return True

        c = getattr(a, "chemistry", None)
        if c is None:
            row = zeros_chemistry()
        else:
            row = chemistry_from_values(c)
        self._chemistry_state[slot, :] = row.astype(np.float32, copy=False)
        self._chemistry_slot_token[slot] = np.int32(token)
        return True

    def _mirror_chemistry_slot_to_agent(self, a) -> bool:
        """Copy the resident chemistry row onto the Agent compatibility mirror."""
        slot = int(getattr(a, "slot", -1))
        if slot < 0 or self._chemistry_state is None or self._chemistry_slot_token is None:
            return False
        if slot >= int(self._chemistry_slot_token.shape[0]):
            return False
        token = int(getattr(a, "slot_token", 0))
        if int(self._chemistry_slot_token[slot]) != token:
            return False

        row = self._chemistry_state[slot]
        cur = getattr(a, "chemistry", None)
        try:
            if getattr(cur, "shape", None) is not None and int(cur.shape[0]) >= CHEMISTRY_SIZE:
                cur[:CHEMISTRY_SIZE] = row
                return True
        except Exception:
            pass
        try:
            a.chemistry = np.asarray(row, dtype=np.float32).copy()
            return True
        except Exception:
            return False

    def _ensure_maintenance_arrays(self, required_slots: int = 0) -> None:
        """Ensure runner-level maintenance arrays exist and match slot capacity."""
        sp = getattr(self.pop, "slot_pool", None)
        if sp is None:
            sp = getattr(self.pop, "_slots", None)

        max_slots = int(getattr(sp, "max_slots", 0) or 0)
        if max_slots <= 0:
            max_slots = int(getattr(self.pop, "max_slots", 0) or 0)
        if max_slots <= 0:
            base = int(getattr(self.pop_cfg, "population_size", 256) or 256)
            factor = int(getattr(self.pop_cfg, "slot_factor", 2) or 2)
            if factor < 2:
                factor = 2
            max_slots = max(1, base * factor)
        if required_slots and required_slots > max_slots:
            max_slots = int(required_slots)

        if self._maintenance_state is not None and int(self._maintenance_state.shape[0]) >= max_slots:
            if self._maintenance_slot_token is not None and int(self._maintenance_slot_token.shape[0]) >= max_slots:
                return

        if self._maintenance_state is None:
            maint = np.tile(zeros_maintenance_state(), (max_slots, 1)).astype(np.float32, copy=False)
        else:
            old_n = int(self._maintenance_state.shape[0])
            maint = np.tile(zeros_maintenance_state(), (max_slots, 1)).astype(np.float32, copy=False)
            maint[: min(old_n, max_slots), :] = self._maintenance_state[: min(old_n, max_slots), :].astype(
                np.float32,
                copy=False,
            )
        self._maintenance_state = maint

        if self._maintenance_slot_token is None or int(self._maintenance_slot_token.shape[0]) < max_slots:
            tok = np.full((max_slots,), -1, dtype=np.int32)
            if self._maintenance_slot_token is not None:
                n = min(int(self._maintenance_slot_token.shape[0]), max_slots)
                tok[:n] = self._maintenance_slot_token[:n].astype(np.int32, copy=False)
            self._maintenance_slot_token = tok

    def _sync_maintenance_slot(self, a, *, force_from_agent: bool = False) -> bool:
        """Initialize or refresh the resident maintenance row for one agent slot."""
        self._ensure_maintenance_arrays()
        slot = int(getattr(a, "slot", -1))
        if slot < 0:
            return False
        if self._maintenance_state is None or self._maintenance_slot_token is None:
            return False
        if slot >= int(self._maintenance_slot_token.shape[0]):
            self._ensure_maintenance_arrays(required_slots=slot + 1)
            if self._maintenance_state is None or self._maintenance_slot_token is None:
                return False
            if slot >= int(self._maintenance_slot_token.shape[0]):
                return False

        token = int(getattr(a, "slot_token", 0))
        if not force_from_agent and int(self._maintenance_slot_token[slot]) == token:
            return True

        m = getattr(a, "maintenance", None)
        if m is None:
            row = zeros_maintenance_state()
        else:
            row = maintenance_from_values(m)
        self._maintenance_state[slot, :] = row.astype(np.float32, copy=False)
        self._maintenance_slot_token[slot] = np.int32(token)
        return True

    def _mirror_maintenance_slot_to_agent(self, a) -> bool:
        """Copy the resident maintenance row onto the Agent compatibility mirror."""
        slot = int(getattr(a, "slot", -1))
        if slot < 0 or self._maintenance_state is None or self._maintenance_slot_token is None:
            return False
        if slot >= int(self._maintenance_slot_token.shape[0]):
            return False
        token = int(getattr(a, "slot_token", 0))
        if int(self._maintenance_slot_token[slot]) != token:
            return False

        row = self._maintenance_state[slot]
        cur = getattr(a, "maintenance", None)
        try:
            if getattr(cur, "shape", None) is not None and int(cur.shape[0]) >= MAINTENANCE_SIZE:
                cur[:MAINTENANCE_SIZE] = row
                return True
        except Exception:
            pass
        try:
            a.maintenance = np.asarray(row, dtype=np.float32).copy()
            return True
        except Exception:
            return False


    def _ensure_identity_arrays(self, required_slots: int = 0) -> None:
        """Ensure runner-level identity drift arrays exist and match slot capacity.

        This is safe to call frequently; it is a no-op once arrays are large enough.

        Args:
            required_slots: Optional minimum capacity demanded by the caller.
        """
        sp = getattr(self.pop, "slot_pool", None)
        if sp is None:
            sp = getattr(self.pop, "_slots", None)

        max_slots = int(getattr(sp, "max_slots", 0) or 0)

        if max_slots <= 0:
            max_slots = int(getattr(self.pop, "max_slots", 0) or 0)

        if max_slots <= 0:
            base = int(getattr(self.pop_cfg, "population_size", 256) or 256)
            factor = int(getattr(self.pop_cfg, "slot_factor", 2) or 2)
            if factor < 2:
                factor = 2
            max_slots = max(1, base * factor)

        if required_slots and required_slots > max_slots:
            max_slots = int(required_slots)

        # Already sufficient
        if self._idprobe_baseline_probs is not None and int(self._idprobe_baseline_probs.shape[0]) >= max_slots:
            if self._idprobe_baseline_token is not None and int(self._idprobe_baseline_token.shape[0]) >= max_slots:
                if self._idprobe_drift_ema is not None and int(self._idprobe_drift_ema.shape[0]) >= max_slots:
                    return

        A = int(getattr(self, "_out", 7) or 7)

        def _grow_2d(old: Optional[np.ndarray], cols: int, fill: float = 0.0) -> np.ndarray:
            if old is None:
                return np.full((max_slots, cols), fill_value=float(fill), dtype=np.float32)
            old_n = int(old.shape[0])
            if old_n >= max_slots:
                return old
            out = np.full((max_slots, cols), fill_value=float(fill), dtype=np.float32)
            out[:old_n, :] = old
            return out

        def _grow_1d(old: Optional[np.ndarray], fill: float = 0.0, dtype=np.float32) -> np.ndarray:
            if old is None:
                return np.full((max_slots,), fill_value=fill, dtype=dtype)
            old_n = int(old.shape[0])
            if old_n >= max_slots:
                return old
            out = np.full((max_slots,), fill_value=fill, dtype=dtype)
            out[:old_n] = old
            return out

        self._idprobe_baseline_probs = _grow_2d(self._idprobe_baseline_probs, cols=A, fill=0.0)
        self._idprobe_baseline_token = _grow_1d(self._idprobe_baseline_token, fill=-1, dtype=np.int32)
        self._idprobe_drift_ema = _grow_1d(self._idprobe_drift_ema, fill=0.0, dtype=np.float32)

    def _sync_drive_slot(self, a) -> None:
        """Ensure slot drive state is initialized for this agent (handles slot reuse).

        Semantics:
        - If the runner has never seen this slot_token, initialize runner drives
          from the agent's mirrored `a.drives` if present, else zeros.
        - If the slot_token matches, do nothing (slot already initialized for this agent).
        - Keeps `a.drives` mirror in sync with runner arrays.
        """
        self._ensure_drive_arrays()

        slot = int(getattr(a, "slot", -1))
        if slot < 0:
            return

        if self._drive_slot_token is None:
            return

        if slot >= int(self._drive_slot_token.shape[0]):
            # If slot pool grew unexpectedly, re-ensure and re-check.
            self._ensure_drive_arrays()
            if slot >= int(self._drive_slot_token.shape[0]):
                return

        token = int(getattr(a, "slot_token", 0))
        if int(self._drive_slot_token[slot]) == token:
            # Slot already initialized for this token, but we must still keep
            # the agent mirror in sync with runner arrays. This is critical
            # for checkpoint save, which serializes agent.drives, not runner arrays.
            d_existing = getattr(a, "drives", None)
            if d_existing is not None:
                try:
                    if getattr(d_existing, "shape", None) is not None and int(d_existing.shape[0]) >= 4:
                        d_existing[0] = self._drive_hunger[slot]
                        d_existing[1] = self._drive_pain[slot]
                        d_existing[2] = self._drive_fatigue[slot]
                        d_existing[3] = self._drive_fear[slot]
                except Exception:
                    pass
                try:
                    if isinstance(d_existing, list) and len(d_existing) >= 4:
                        d_existing[0] = float(self._drive_hunger[slot])
                        d_existing[1] = float(self._drive_pain[slot])
                        d_existing[2] = float(self._drive_fatigue[slot])
                        d_existing[3] = float(self._drive_fear[slot])
                except Exception:
                    pass
            return

        d = getattr(a, "drives", None)

        def _init_from_agent_or_zero(idx: int) -> np.float32:
            if d is None:
                return np.float32(0.0)
            try:
                if getattr(d, "shape", None) is not None and int(d.shape[0]) >= 4:
                    return np.float32(float(d[idx]))
            except Exception:
                pass
            # Also accept plain Python sequences (e.g. drives loaded from JSON)
            try:
                if isinstance(d, (list, tuple)) and len(d) >= 4:
                    return np.float32(float(d[idx]))
            except Exception:
                pass
            return np.float32(0.0)

        # Initialize runner arrays
        self._drive_hunger[slot] = _init_from_agent_or_zero(0)
        self._drive_pain[slot] = _init_from_agent_or_zero(1)
        self._drive_fatigue[slot] = _init_from_agent_or_zero(2)
        self._drive_fear[slot] = _init_from_agent_or_zero(3)

        self._drive_slot_token[slot] = np.int32(token)

        # Keep agent mirror in sync (best-effort)
        if d is not None:
            try:
                if getattr(d, "shape", None) is not None and int(d.shape[0]) >= 4:
                    d[0] = self._drive_hunger[slot]
                    d[1] = self._drive_pain[slot]
                    d[2] = self._drive_fatigue[slot]
                    d[3] = self._drive_fear[slot]
            except Exception:
                pass
            try:
                if isinstance(d, list) and len(d) >= 4:
                    d[0] = float(self._drive_hunger[slot])
                    d[1] = float(self._drive_pain[slot])
                    d[2] = float(self._drive_fatigue[slot])
                    d[3] = float(self._drive_fear[slot])
            except Exception:
                pass

    def _phase3_pre_start_generation_sync(self, alive_agents) -> None:
        """Phase 3 determinism: make pre-generation transient state symmetric.

        Fresh runs and resume-from-checkpoint runs can differ in *when* certain
        transient-but-semantics-affecting structures become initialized.

        Examples:
        - Runner drive arrays (per-slot) are initialized lazily via _sync_drive_slot.
        - Policy parameter banks and GRU memory state (_Hmem) are initialized lazily
          during bank sync.

        If resume performs these initializations earlier (during resume-apply), while
        fresh does them later (first policy tick), state captured at the
        "pre_start_generation_alive" probe boundary diverges and can cascade into
        action differences.

        This helper forces an idempotent, deterministic initialization before we emit
        the pre-start-generation probe.

        Constraints:
        - Must not draw RNG.
        - Must be safe to call every generation.
        - Must be safe for both fresh and resumed runs.
        """
        try:
            agents = list(alive_agents) if alive_agents is not None else []
        except Exception:
            agents = []

        # 1) Ensure drive arrays exist and are initialized for all alive agents.
        try:
            self._ensure_drive_arrays()
            for a in agents:
                try:
                    self._sync_drive_slot(a)
                except Exception:
                    continue
        except Exception:
            pass

        # 2) Ensure parameter banks are synced (MLP + GRU). This also ensures GRU
        #    memory state exists and reset slots are deterministic.
        try:
            if hasattr(self, "_sync_bank_for_agents"):
                self._sync_bank_for_agents(agents, buckets=None)
            if hasattr(self, "_sync_gru_bank_for_agents"):
                self._sync_gru_bank_for_agents(agents, buckets=None)
        except Exception:
            pass
