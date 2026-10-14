from __future__ import annotations

"""
simulation/runner_pkg/runner_config.py

Simulation configuration dataclass.

Goals
-----
- Keep configuration flat and easy to serialize.
- Keep defaults safe (features OFF unless explicitly enabled).
- Provide knobs for selection pressure without refactoring other systems.

Episode-0 coherence selection pressure (Step 2 split-knobs)
----------------------------------------------------------
We support two modes:

A) Legacy / combined mode (backward compatible):
    effective_fitness += episode0_fitness_weight * episode0_term
   Where episode0_term is a weighted sum of normalized episode-0 signals.

B) Split-knob mode (preferred for causal tests):
    effective_fitness +=
        episode0_pred_err_fitness_weight      * ep0_pred_term
      + episode0_thrashing_fitness_weight     * ep0_thrash_term
      + episode0_drive_jitter_fitness_weight  * ep0_jitter_term
      + episode0_selfharm_fitness_weight      * ep0_selfharm_term

Split mode is enabled automatically if ANY of the split weights is non-zero.
Defaults are 0.0 so nothing changes unless you explicitly set them.

Episode-0 signals used (existing metrics only):
- episode0_pred_err_mean
- episode0_drive_jitter_mean
- episode0_switches_rate / episode0_oscillations_rate
- episode0_selfharm_rate

Step 3: Developmental coupling analysis (NO mechanics)
------------------------------------------------------
This adds an optional *analysis-only* generation-end log:
- devcoupling.csv

It bins agents by episode-0 prediction error and compares downstream outcomes:
- later-life prediction error (derived from existing sums/counts)
- injury
- reproduction
- survival

This is fully deterministic and does not affect selection unless you later
decide to add a gated conditioning experiment.

Option C: Late-life coherence tax (minimal world pressure, gated)
-----------------------------------------------------------------
This is the minimal “make later-life cognition matter” lever.

If enabled, after a configurable post-episode-0 start step, agents pay a tiny
extra per-step energy cost proportional to their *later-life* prediction error.

Key properties:
- No new agent behaviors
- No new learning algorithm
- No new world events
- Deterministic given the same seed
- OFF by default (safe)

This is intended to test whether evolution starts to care about adult predictive
coherence once confusion becomes slightly costly *later*, not just at birth.
"""

from dataclasses import dataclass, field

from evolution.inheritance_registry_v1 import LEGACY_INHERITANCE_PROFILE

from evolution.structural_development_v1 import (
    CYCLE_POLICY,
    DEPENDENCY_IDENTITY_AUTHORITY,
    DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
    DEVELOPMENTAL_STAGE_IDS,
    DEVELOPMENTAL_STAGE_PROFILE_VERSION,
    DEVELOPMENT_OPERATOR_ATTEMPTS,
    DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT,
    DEVELOPMENT_OPERATOR_PROFILE,
    INACTIVE_OUTPUT_POLICY,
    INACTIVE_STATE_POLICY,
    LATCH_SEMANTICS,
    LATCH_VERSION,
    LATE_START_STEP,
    MATURE_START_STEP,
    MAX_DEPENDENCY_EDGES_PER_AGENT,
    MAX_PREREQUISITES_PER_MODULE,
    MAX_STAGE_GAIN,
    MIGRATION_RNG_DRAWS,
    MIN_STAGE_GAIN,
    SUPPORTED_LANE,
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
    GATED_PROMOTER_POLICY_VERSION,
    GATED_RECORD_SCHEMA_VERSION,
    GATED_RUNTIME_SCHEMA_VERSION,
    GATED_STATE_WIDTH,
    GATED_SUPPORTED_LANE,
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


@dataclass
class SimConfig:
    generations: int = 50
    steps_per_generation: int = 150
    seed: int = 0

    # Execution authority is explicit.  Legacy is deliberately the default so
    # established G/T routes retain their frozen qualification semantics.
    execution_route_identity: str = "legacy_generation_route_v1"
    total_ticks: int | None = None
    runtime_partition_ticks: int | None = None
    report_every_ticks: int = 0
    checkpoint_every_ticks: int = 0

    # TN-2 observation is a passive projection after committed ticks.
    # None/0 is OFF; a positive value samples each such tick interval.
    science_sample_interval_ticks: int | None = None

    # -----------------------------
    # Phase 2: ControlPlane identity (per-branch sequence numbers)
    # -----------------------------
    # Optional identifiers; used for ControlPlane ordering persistence and operator tooling.
    branch_id: str = "default"
    campaign_id: str = ""

    # RECOMB-1 inheritance mechanics are explicitly profiled.  Legacy remains
    # the default so existing campaigns and checkpoint identifiers are
    # unchanged unless the repaired profile is deliberately selected.
    inheritance_profile: str = LEGACY_INHERITANCE_PROFILE
    diploid_mutation_enable: bool = True
    genome_upgrade_v2_enable: bool = False
    genome_upgrade_v2_mutation_rate_q10: int = 20
    genome_upgrade_v2_mutation_step_q10: int = 64
    genome_upgrade_v2_founder_profile_id: str = "neutral"
    genome_upgrade_v2_founder_profile_sha256: str = ""
    genome_v2_causal_capture_enable: bool = False
    genome_v2_checkpoint_authority_binding: tuple[tuple[str, str], ...] = ()


    # -----------------------------
    # Phase 5B: Trait Injection Framework
    # -----------------------------
    # Campaign-frozen apply phase for all TraitInjectionCommand interventions.
    # Allowed values:
    # - post-reproduction-pre-selection
    # - post-selection-pre-mutation
    trait_injection_apply_phase: str = "post-selection-pre-mutation"


    # -----------------------------
    # Device / execution controls
    # -----------------------------
    # Explicit named implementation profile applied by CLI/wrappers. Profiles
    # are default-off and persist their audit into run_config for evidence.
    performance_profile: str = ""
    performance_profile_audit: dict | None = None

    # Campaign output/storage policy. These fields are diagnostics/output only:
    # they are persisted for manifests and harness preflight, and do not alter
    # selection, reproduction, rewards, cognition, chemistry, lineage mechanics,
    # Patch/Jungle mechanics, WDL mechanics, or RNG behavior.
    campaign_output_profile: str = "compact_default"
    allow_large_output_profile: bool = False
    allow_generation_checkpointing: bool = False

    # Exact Architecture B CUDA run owner. Production admission is bounded to
    # Patch P2/g4/t80/population-64/grid-16 and asexual B1-B64.
    cuda_architecture_b_production_enable: bool = True
    cuda_architecture_b_max_births_per_tick: int = 64
    cuda_architecture_b_request_mode: str = "auto"
    cuda_architecture_b_allow_explicit_exactness_override: bool = False
    cuda_architecture_b_artifact_path: str = ""
    cuda_architecture_b_qualification_receipt_path: str = ""
    cuda_architecture_b_reporting_receipt_path: str = ""
    cuda_architecture_b_qualification_registry_path: str = ""
    cuda_architecture_b_promotion_registry_path: str = ""
    cuda_architecture_b_performance_decision: str = "unmeasured"
    cuda_architecture_b_performance_decision_measured: bool = False
    cuda_architecture_b_cpython_gauss_segment_capacity: int = -1
    cuda_architecture_b_policy_birth_segment_refill_enable: bool = False
    cuda_architecture_b_policy_birth_segment_capacity: int = -1
    cuda_architecture_b_turnover_mlp_shadow_elide_enable: bool = False
    cuda_architecture_b_cpu_request_staging_enable: bool = False

    # Scientific Chronicle V1 is a read-only, RNG-free observer. It is OFF by
    # default and cannot feed events or rendered text back into mechanics.
    chronicle_enable: bool = False
    chronicle_profile: str = "notable"
    chronicle_run_id: str = ""
    chronicle_planned_generations: int = 0
    chronicle_terminal_enable: bool = False
    chronicle_episode_enable: bool = True
    chronicle_context_window_enable: bool = True
    chronicle_context_events_before: int = 8
    chronicle_context_events_after: int = 4
    chronicle_max_context_windows_per_generation: int = 16
    chronicle_max_live_messages_per_generation: int = 8
    chronicle_max_stored_events_per_generation: int = 32
    chronicle_max_events_per_category_per_generation: int = 12
    chronicle_subject_cooldown_generations: int = 1
    chronicle_region_cooldown_generations: int = 1
    chronicle_min_magnitude: float = 0.0
    chronicle_minimum_persistence_generations: int = 1
    chronicle_aggregation_window_generations: int = 1
    chronicle_episode_gap_generations: int = 1
    chronicle_dominance_threshold: float = 0.50
    chronicle_rapid_lineage_share_delta: float = 0.15
    chronicle_near_extinction_count: int = 1
    chronicle_food_depleted_fraction_threshold: float = 0.25
    chronicle_food_recovery_fraction_threshold: float = 0.10
    chronicle_crowding_hotspot_fraction_threshold: float = 0.20
    chronicle_waste_hotspot_threshold: float = 0.25
    chronicle_bottleneck_survival_fraction: float = 0.50
    chronicle_estimated_event_bytes: int = 2048
    chronicle_schema_version: str = "eiel_formal_event_v1"
    chronicle_episode_schema_version: str = "eiel_scientific_episode_v1"
    chronicle_comparison_schema_version: str = "eiel_scientific_chronicle_comparison_v1"
    # GEN1-F1 is a passive canonical birth/carrier observer. It remains
    # independently default-off so diagnostic runners preserve legacy output.
    gen1_f1_capture_enable: bool = False

    # RECOMB-72 is an audit-only neutral inheritance fixture. It is deliberately
    # default-off so ordinary reproduction, selection, and RNG schedules remain
    # byte-for-byte unchanged. Admission is restricted to w16p64/G1-G30/T80.
    recomb72_audit_enable: bool = False
    recomb72_inheritance_mode: str = "current_authoritative"
    recomb72_founder_pattern: str = "single_extreme_scalar"
    recomb72_audit_seed: int = 0
    recomb72_disable_ordinary_tick_births: bool = True
    recomb72_mutation_off: bool = True
    recomb72_neutral_actions_enable: bool = False
    recomb72_neutral_action_id: int = 5
    recomb72_neutral_environment_enable: bool = False

    # MEM-MAX diagnostic-only pre-cue tick-boundary snapshots. OFF by default.
    # When enabled, the delayed-choice substrate captures an isolated runner
    # clone immediately before cue/selected-target treatment is applied.
    mem_max_pre_cue_checkpoint_enable: bool = False
    mem_max_pre_cue_checkpoint_max_snapshots: int = 1
    mem_max_pre_cue_checkpoint_dir: str = ""

    # "auto" keeps legacy behavior: use CUDA if available, else CPU.
    device: str = "auto"

    # Separate device for the parameter bank (rarely needed).
    # "auto" => same as `device`.
    bank_device: str = "auto"

    # Torch CPU thread count override.
    # 0 => keep runner default (usually cpu_count-1).
    torch_threads: int = 0

    # Optional compilation for policy forward passes (torch.compile).
    # Default OFF for stability.
    torch_compile_policy: bool = False

    # -----------------------------
    # Policy tick stride (multi-tick batching)
    # -----------------------------
    # 1 => legacy behavior (policy called every tick)
    # K>1 => policy called once per K ticks (MLP: K actions cached; GRU: macro-action repeated K ticks)
    policy_tick_stride: int = 2

    # Optional policy-weight variation at offspring construction. Default OFF.
    # When enabled, child policy parameters receive zero-mean Gaussian mutation
    # from the mutation torch RNG after normal inheritance/recombination. This is
    # general policy evolvability support: it is not memory-specific, cue-specific,
    # target-specific, or fitness/reward shaping.
    policy_weight_mutation_enable: bool = False
    policy_weight_mutation_mode: str = "dense"
    policy_weight_mutation_rate: float = 1.0
    policy_weight_mutation_std: float = 0.0

    # Optional exact founder-bank startup loader. Default OFF. When enabled,
    # the runner loads a verified founder artifact after random population
    # construction and before first placement/action, then emits a startup
    # sentinel. This is reporting/startup control only; normal selection,
    # reproduction, and mutation continue after startup.
    founder_bank_artifact_enable: bool = False
    founder_bank_artifact_path: str = ""
    founder_bank_artifact_expected_hash: str = ""
    founder_bank_startup_sentinel_enable: bool = False
    founder_bank_startup_sentinel_filename: str = "founder_bank_startup_sentinel.json"

    # Optional compact policy-lineage telemetry for diagnostic runs. Default
    # OFF. When enabled, child construction records parent/donor/pre-mutation/
    # post-mutation/final policy hashes, tensor delta norms, and bounded
    # cue-action probes without affecting reproduction, selection, rewards, or
    # RNG draws.
    policy_lineage_telemetry_enable: bool = False
    policy_lineage_telemetry_sample_limit: int = 64

    # CORRIDOR-CP0 central-place reproduction eligibility. Default OFF; when
    # enabled with env.corridor_cp0_enable, parents must be in home-local cells.
    corridor_cp0_central_place_eligibility_enable: bool = False
    corridor_cp0_probe_enable: bool = False
    corridor_cp0_probe_schema_version: int = 1
    corridor_cp0_probe_arm: str = ""
    corridor_cp0_probe_controller_version: str = "corridor_cp0_mlp_v1"
    corridor_cp0_telemetry_enable: bool = False
    corridor_cp0_telemetry_schema_version: int = 1
    habacclim1_probe_enable: bool = False
    habacclim1_probe_schema_version: int = 1
    habacclim1_probe_arm: str = ""
    habacclim1_probe_controller_version: str = "habacclim1_mlp_v1"
    habacclim1_telemetry_enable: bool = False
    habacclim1_telemetry_schema_version: int = 1



    # -----------------------------
    # Optional torch.profiler
    # -----------------------------
    enable_profiler: bool = False
    profiler_wait: int = 2
    profiler_warmup: int = 2
    profiler_active: int = 5
    profiler_repeat: int = 1
    profiler_record_shapes: bool = True
    profiler_profile_memory: bool = True
    profiler_with_stack: bool = True

    # Optional performance diagnostic only. When true and CUDA is available,
    # selected perf timers synchronize CUDA before reading wall clocks so scoped
    # GPU work is easier to interpret. Default OFF to avoid normal-run overhead.
    perf_sync_cuda: bool = False

    # ESCAP-1 capacity-epoch checkpoint metadata scaffold. Default OFF; it does
    # not resize arrays or alter runtime behavior.
    capacity_epoch_metadata_enable: bool = False
    capacity_epoch_id: str = "capacity-epoch-0"
    capacity_epoch_sequence: int = 0
    capacity_epoch_parent_id: str = ""
    capacity_epoch_transition_state: str = "monitoring"
    capacity_epoch_manager_enable: bool = False

    # Optional performance prototype only. When enabled, the policy_tick_stride
    # k>1 path packs observations, scalar policy inputs, drives, and drive gains
    # into one staging tensor before transfer. Default OFF preserves the current
    # staging path until same-seed parity and benchmark evidence justify adoption.
    perf_policy_drive_packed_k_staging_enable: bool = False

    # Optional AGGPERF-3 prototype only. When enabled, the policy_tick_stride
    # cache uses slot-indexed storage with agent_id/token guards instead of
    # per-agent dict lookups. Default OFF preserves the current cache path.
    perf_policy_stride_slot_cache_enable: bool = False

    # Optional AGGPERF-3 prototype only. When enabled, skip the deterministic
    # pre-policy developmental-expression warmup pass and let policy staging
    # populate the same tick-local cache on first use. Default OFF preserves
    # trace ordering and the current conservative determinism path.
    perf_skip_dev_expression_prewarm_enable: bool = False

    # Optional low-overhead AGGPERF-3 bucket grouping. Default OFF; when enabled
    # it adds coarse policy/env/diagnostic/internal timing buckets to the normal
    # perf_buckets.csv stream.
    perf_aggperf3_timing_buckets_enable: bool = False

    # Optional CXXREWRITE-0 array-backend ownership scaffold. Default OFF.
    # Shadow mode copies current Python/env-authoritative live state into
    # slot-indexed arrays at generation boundaries and validates the copy. It
    # does not mutate behavior. Authoritative mode intentionally fails closed
    # until churn tape and sync/writeback ownership exist.
    runner_array_backend_enable: bool = False
    cxxrewrite_array_backend_shadow_enable: bool = False
    cxxrewrite_array_backend_authoritative_enable: bool = False
    cxxrewrite_array_backend_debug_sync_enable: bool = False
    cxxrewrite_array_backend_fail_fast: bool = False

    # Optional CXXREWRITE-1 array lifecycle gate. Default OFF. This mode
    # replays compact generation-boundary churn into mutable slot arrays and
    # can sync the boundary state back to Python/env objects. It is lifecycle
    # ownership scaffolding only; native kernels remain out of scope here.
    cxxrewrite_array_backend_enable: bool = False
    cxxrewrite_lifecycle_shadow_enable: bool = False
    cxxrewrite_lifecycle_authoritative_enable: bool = False
    cxxrewrite_lifecycle_fail_closed: bool = False
    cxxrewrite_lifecycle_sync_python_objects_enable: bool = False
    cxxrewrite_lifecycle_record_churn_tape_enable: bool = False
    cxxrewrite_lifecycle_validate_shadow_enable: bool = False

    # Optional CXXREWRITE-2 native tick/env-apply kernel gate. Default OFF.
    # Shadow mode compares a native no-RNG movement/gather/action-cost slice
    # against the Python env step. Authoritative native mutation intentionally
    # fails closed until broader Python-owned tick semantics are migrated.
    cxxrewrite_native_tick_enable: bool = False
    cxxrewrite_native_tick_shadow_enable: bool = False
    cxxrewrite_native_tick_authoritative_enable: bool = False
    cxxrewrite_native_tick_fail_closed: bool = False
    cxxrewrite_native_tick_validate_shadow_enable: bool = False
    cxxrewrite_native_tick_compare_enable: bool = True
    cxxrewrite_native_tick_library_path: str = ""
    cxxrewrite_native_tick_write_summary: bool = True
    cxxrewrite_native_tick_summary_filename: str = "cxxrewrite2_native_tick_summary.json"
    cxxrewrite_native_observation_enable: bool = False
    cxxrewrite_native_policy_bridge_enable: bool = False

    # Optional CXXREWRITE-3 resident PolicyFrame/native-entry packing gate.
    # Default OFF. Shadow/native-packing modes update resident arrays from
    # already-materialized tick arrays before CXXREWRITE2 native entry, so the
    # native path can consume current resident state without repairing stale
    # scalars inside native packing. Authoritative modes intentionally fail
    # closed until broader runner ownership is migrated.
    cxxrewrite_resident_frame_enable: bool = False
    cxxrewrite_resident_frame_shadow_enable: bool = False
    cxxrewrite_resident_frame_authoritative_enable: bool = False
    cxxrewrite_resident_frame_fail_closed: bool = False
    cxxrewrite_resident_frame_validate_shadow_enable: bool = False
    cxxrewrite_resident_frame_native_packing_enable: bool = False
    cxxrewrite_resident_frame_native_packing_shadow_enable: bool = False
    cxxrewrite_resident_frame_native_packing_authoritative_enable: bool = False
    cxxrewrite_resident_frame_native_packing_compare_enable: bool = True
    cxxrewrite_resident_frame_native_packing_library_path: str = ""
    cxxrewrite_resident_frame_sync_python_boundary_enable: bool = False
    cxxrewrite_resident_frame_forbid_per_tick_agent_sync: bool = False
    cxxrewrite_resident_frame_write_summary: bool = True
    cxxrewrite_resident_frame_summary_filename: str = "cxxrewrite3_resident_frame_summary.json"

    # Optional CXXREWRITE-4 resident upstream policy/observation scalar gate.
    # Default OFF. Enabled modes read policy scalar heads from the resident frame,
    # mirror post-env scalar mutations back into resident arrays, and can attach a
    # native policy-head pack helper for parity/timing. Authoritative modes fail
    # closed in this gate because policy execution, reproduction, metrics, and
    # checkpoint semantics are not fully resident-array owned yet.
    cxxrewrite_resident_observation_enable: bool = False
    cxxrewrite_resident_observation_shadow_enable: bool = False
    cxxrewrite_resident_observation_authoritative_enable: bool = False
    cxxrewrite_resident_observation_native_pack_enable: bool = False
    cxxrewrite_resident_observation_native_pack_library_path: str = ""
    cxxrewrite_resident_policyframe_enable: bool = False
    cxxrewrite_resident_policyframe_shadow_enable: bool = False
    cxxrewrite_resident_policyframe_authoritative_enable: bool = False
    cxxrewrite_resident_policyframe_native_pack_enable: bool = False
    cxxrewrite_resident_policyframe_native_pack_library_path: str = ""
    cxxrewrite_resident_post_env_scalar_enable: bool = False
    cxxrewrite_resident_post_env_scalar_shadow_enable: bool = False
    cxxrewrite_resident_post_env_scalar_authoritative_enable: bool = False
    cxxrewrite_resident_post_env_scalar_fail_closed: bool = False
    cxxrewrite_resident_upstream_validate_shadow_enable: bool = False
    cxxrewrite_resident_upstream_forbid_hotpath_agent_sync: bool = False
    cxxrewrite_resident_upstream_write_summary: bool = True
    cxxrewrite_resident_upstream_defer_summary_write_enable: bool = False
    cxxrewrite_resident_upstream_summary_filename: str = "cxxrewrite4_upstream_summary.json"

    # Optional resident-generation runner fast loop. Default OFF. This is a
    # guarded live-run performance path that keeps Python subsystem mechanics
    # authoritative but removes the default debug/probe shell and builds the
    # per-tick alive surface from resident env/pop arrays. It falls back unless
    # diagnostics/native shadow gates that rely on the legacy shell are off.
    cxxrewrite_resident_generation_runner_enable: bool = False
    cxxrewrite_resident_generation_runner_fail_closed: bool = False
    cxxrewrite_resident_generation_runner_write_summary: bool = True
    cxxrewrite_resident_generation_runner_summary_filename: str = "cxxrewrite_resident_generation_runner_summary.json"
    cxxrewrite_resident_generation_scalar_env_inputs_enable: bool = False
    cxxrewrite_resident_generation_chemistry_arrays_enable: bool = False
    cxxrewrite_resident_generation_maintenance_arrays_enable: bool = False

    # Optional resident developmental scalar cache. Default OFF. When enabled,
    # the authoritative developmental-expression resolver still runs once per
    # live agent/tick, but policy/drives/reproduction can reuse compact scalar
    # fields instead of repeating cache-hit resolver and dict extraction work.
    cxxrewrite_resident_dev_scalar_cache_enable: bool = False

    # Optional resident developmental stage-expression cache. Default OFF.
    # Reuses a fully resolved expression across ticks while the agent remains in
    # the same developmental stage and the genome/program/module objects are
    # unchanged. Stage transitions still go through the authoritative resolver.
    cxxrewrite_resident_dev_stage_cache_enable: bool = False

    # Optional resident developmental stage-bounds cache. Default OFF. When the
    # stage-expression cache is active, store the current stage interval and
    # reuse it to avoid recomputing developmental_stage_id while the agent age
    # remains inside the same canonical stage window.
    cxxrewrite_resident_dev_stage_bounds_cache_enable: bool = False

    # Optional reproduction hotpath pass-through. Default OFF. When C2 and C1
    # are both enabled, feed the merged C2 payload directly into the C1 canonical
    # boundary instead of rehydrating an intermediate Genome only to serialize it
    # again. The final child genome is still built through Genome.from_dict.
    cxxrewrite_reproduction_c2_payload_passthrough_enable: bool = False

    # Optional reproduction C1 fast path. Default OFF. When C2 payload
    # pass-through is active, C1 may trust the already-canonical C2
    # modules/connections for its first operator input. It still canonicalizes
    # after each C1 operator and before Genome.from_dict.
    cxxrewrite_reproduction_c1_trusted_c2_payload_enable: bool = False

    # Optional reproduction C2 parent fast path. Default OFF. C2 may consume
    # cached canonical parent Genome payloads instead of re-canonicalizing both
    # parents at each birth. The trusted path fails closed if the payload shape
    # is not already canonical.
    cxxrewrite_reproduction_c2_trusted_parent_payloads_enable: bool = False

    # Optional reproduction materialization fast path. Default OFF. In the
    # trusted C2->C1 payload pass-through stack, build the intermediate child
    # raw payload directly instead of instantiating a temporary Genome that is
    # immediately serialized back into a payload for C2/C1.
    cxxrewrite_reproduction_direct_child_payload_enable: bool = False

    # Optional reproduction C1 canonical-boundary fast paths. Default OFF.
    # These are legacy-loop equivalents of bounded RUNNERCORE C1 slices: they
    # only activate after C2 payload pass-through and trusted C1 input are
    # already active, preserving RNG/operator accounting while avoiding
    # redundant canonical normalization work in child genome construction.
    cxxrewrite_reproduction_c1_final_normalize_skip_enable: bool = False
    cxxrewrite_reproduction_c1_noop_normalize_skip_enable: bool = False
    cxxrewrite_reproduction_c1_empty_structure_fastpath_enable: bool = False
    cxxrewrite_reproduction_c1_empty_structure_fastpath_fail_closed: bool = False

    # Optional reproduction policy crossover fast path. Default OFF. When MLP
    # parent/child parameter tensors are all on one device, draw one flat mask
    # tensor and slice it across parameters instead of issuing one torch.rand per
    # parameter tensor. The total RNG draw count and element order are preserved.
    cxxrewrite_reproduction_flat_policy_crossover_enable: bool = False

    # Optional RUNNERCORE-0 supported-profile runner contract. Default OFF.
    # This is a parity-first boundary gate for fast_exploratory_metrics_only_v2.
    # Shadow mode imports resident slot arrays for diagnostics only. Candidate,
    # authoritative, and native modes intentionally fail closed until the full
    # tick/action/churn/metrics/checkpoint lifecycle is resident-array owned.
    runnercore_supported_profile_enable: bool = False
    runnercore_supported_profile: str = "fast_exploratory_metrics_only_v2"
    runnercore_shadow_contract_enable: bool = False
    runnercore_candidate_enable: bool = False
    runnercore_authoritative_enable: bool = False
    runnercore_native_enable: bool = False
    runnercore_legacy_compare_enable: bool = False
    runnercore_fail_closed_on_unsupported: bool = True
    runnercore_export_sync_mode: str = "generation_boundary"
    runnercore_write_summary: bool = True
    runnercore_summary_filename: str = "runnercore0_supported_profile_summary.json"

    # Optional CXXWAVE1F scalar C++ backbone. Default OFF.
    # When used inside RunnerCore, this replaces the Python drive, chemistry,
    # and maintenance scalar loops only for a strict support envelope. It is
    # fail-closed for positive prediction horizons and maintenance opportunity
    # grid scans until those surfaces are native-owned too.
    cxxwave1f_scalar_backbone_enable: bool = False
    cxxwave1f_scalar_backbone_fail_closed: bool = True
    cxxwave1f_scalar_backbone_library_path: str = ""
    cxxwave1f_scalar_backbone_write_summary: bool = True
    cxxwave1f_scalar_backbone_summary_filename: str = "cxxwave1f_cpp_backbone_summary.json"
    cxxwave1g_full_cpp_backbone_enable: bool = False
    cxxwave1g_full_cpp_backbone_fail_closed: bool = True
    cxxwave1g_full_cpp_backbone_write_summary: bool = True
    cxxwave1g_full_cpp_backbone_summary_filename: str = "cxxwave1g_full_live_cpp_backbone_summary.json"
    cxxbackbone2_live_composition_enable: bool = False
    cxxbackbone2_live_composition_fail_closed: bool = True
    cxxbackbone2_live_composition_write_summary: bool = True
    cxxbackbone2_live_composition_summary_filename: str = "cxxbackbone2_live_composition_summary.json"
    cxxbackbone2_policy_core_live_probe_enable: bool = False
    cxxbackbone2_policy_core_live_probe_write_summary: bool = True
    cxxbackbone2_policy_core_live_probe_summary_filename: str = "cxxbackbone2_policy_core_live_probe_summary.json"
    cxxbackbone2_policy_core_live_probe_build_dir: str = ""
    cxxbackbone3_policy_aten_gru_authority_enable: bool = False
    cxxbackbone4_policy_aten_mlp_authority_enable: bool = False
    cxxbackbone3_policy_aten_build_dir: str = ""
    cxxbackbone5_policy_envelope_profile_enable: bool = False
    cxxbackbone5_policy_envelope_fail_closed: bool = True
    cxxbackbone6_resident_policy_vertical_slice_enable: bool = False
    cxxbackbone6_resident_policy_vertical_slice_fail_closed: bool = True
    cxxbackbone7_resident_patch_policy_action_loop_enable: bool = False
    cxxbackbone7_resident_patch_policy_action_loop_fail_closed: bool = True
    cxxbackbone8_resident_flat_patch_generation_island_enable: bool = False
    cxxbackbone8_resident_flat_patch_generation_island_fail_closed: bool = True
    cxxbackbone8_resident_flat_patch_generation_island_write_summary: bool = True
    cxxbackbone8_resident_flat_patch_generation_island_summary_filename: str = (
        "cxxbackbone8_generation_island_summary.json"
    )
    cxxbackbone8r_flat_cpu_native_action_apply_enable: bool = False
    cxxbackbone8r_flat_cpu_native_policy_inputs_enable: bool = False
    cxxbackbone8r_patch_cpu_native_action_apply_enable: bool = False
    cxxbackbone8r_patch_cpu_native_policy_inputs_enable: bool = False
    cxxbackbone13_owner_generation_runner_enable: bool = False
    cxxbackbone13_owner_generation_runner_fail_closed: bool = True
    cxxbackbone13_owner_generation_runner_write_summary: bool = True
    cxxbackbone13_owner_generation_runner_summary_filename: str = (
        "cxxbackbone13_owner_generation_runner_summary.json"
    )
    native_execution_profile: str = "native_safe_default"
    native_safe_default_enable: bool = True
    native_exact_multigen_required: bool = True
    native_runtime_diagnostic_enable: bool = False
    native_runtime_diagnostic_ack_multigen_parity_open: bool = False
    native_runtime_enable: bool = False
    native_runtime_fail_closed: bool = True
    native_runtime_birth_commit_enable: bool = True
    native_runtime_max_births_per_generation: int = 1
    # Phase E passive modular checkpoint ownership. This remains storage-only:
    # no module execution, structural mutation, birth, or slot lifecycle is
    # permitted while enabled.
    native_runtime_modular_storage_enable: bool = False
    native_runtime_build_dir: str = ""
    native_runtime_allow_runtime_jit_build: bool = False
    native_runtime_fixed_policy_family: str = ""
    native_runtime_cxx26_scope_lock_enable: bool = False
    native_runtime_write_summary: bool = True
    # Opt-in shared reference/native lifecycle serialization. The legacy native-
    # named stream remains the default until route-neutral output is requested.
    typed_lifecycle_public_output_route_neutral_enable: bool = False

    # Optional RUNNERCORE-1 array-owned tick authority lane. Default OFF.
    # Candidate mode replaces the live tick producer for a narrow supported
    # fast profile: resident arrays own policy inputs, stride action cache,
    # GRU Hmem updates, env scalar results, common tick stats, drives, and
    # chemistry until a defined boundary export. Unsupported churn/checkpoint
    # surfaces fail closed instead of falling back to legacy Agent authority.
    runnercore1_array_tick_authority_enable: bool = False
    runnercore1_supported_profile: str = "supported_fast_runner_v1"
    runnercore1_candidate_enable: bool = False
    runnercore1_authoritative_enable: bool = False
    runnercore1_native_enable: bool = False
    runnercore1_fail_closed_on_unsupported: bool = True
    runnercore1_allow_generation_boundary_export: bool = True
    runnercore1_allow_churn_boundary_materialization: bool = False
    runnercore1_write_summary: bool = True
    runnercore1_summary_filename: str = "runnercore1_array_tick_authority_summary.json"

    # Optional RUNNERCORE-2 controlled churn slot lifecycle lane. Default OFF.
    # Extends RUNNERCORE-1's resident array authority to a bounded no-mate birth
    # profile with explicit event-boundary Agent materialization for canonical
    # child genome/policy construction, then immediately re-attaches newborn
    # rows, policy banks, Hmem rows, slot tokens, and action-cache ownership to
    # the resident arrays. Unsupported churn surfaces fail closed.
    runnercore2_churn_slot_lifecycle_enable: bool = False
    runnercore2_supported_profile: str = "supported_fast_runner_v2"
    runnercore2_candidate_enable: bool = False
    runnercore2_authoritative_enable: bool = False
    runnercore2_native_enable: bool = False
    runnercore2_fail_closed_on_unsupported: bool = True
    runnercore2_allow_birth_boundary_agent_materialization: bool = False
    runnercore2_write_summary: bool = True
    runnercore2_summary_filename: str = "runnercore2_churn_slot_lifecycle_summary.json"

    # Optional RUNNERCORE-3 profile-guided churn acceleration lane. Default OFF.
    # RUNNERCORE-3 extends the accepted RUNNERCORE-2 v2 churn lane with
    # profiling and one bounded candidate acceleration selected from live-run
    # hotspots. It remains fail-closed and must not reintroduce per-tick Agent
    # synchronization or legacy action materialization.
    runnercore3_profile_enable: bool = False
    runnercore3_candidate_enable: bool = False
    runnercore3_native_enable: bool = False
    runnercore3_fail_closed_on_unsupported: bool = True
    runnercore3_allow_birth_boundary_agent_materialization: bool = False
    runnercore3_vectorized_birth_precheck_enable: bool = False
    runnercore3_fast_birth_event_boundary_enable: bool = False
    runnercore3_self_recombination_fastpath_enable: bool = False
    runnercore3_vectorized_action_cache_enable: bool = False
    runnercore3_write_summary: bool = True
    runnercore3_summary_filename: str = "runnercore3_profile_guided_native_churn_summary.json"

    # Optional RUNNERCORE-4 structural payload fastpath. Default OFF.
    # RUNNERCORE-4 remains inside the accepted supported_fast_runner_v2 churn
    # lane and only accelerates no-mate self reproduction child construction
    # when explicitly parity-gated by the harness.
    runnercore4_profile_enable: bool = False
    runnercore4_candidate_enable: bool = False
    runnercore4_native_enable: bool = False
    runnercore4_fail_closed_on_unsupported: bool = True
    runnercore4_allow_birth_boundary_agent_materialization: bool = False
    runnercore4_structural_payload_cache_enable: bool = False
    runnercore4_changed_parent_import_enable: bool = False
    runnercore4_write_summary: bool = True
    runnercore4_summary_filename: str = "runnercore4_structural_payload_fastpath_summary.json"

    # Optional RUNNERCORE-5 child-construction ownership lane. Default OFF.
    # RUNNERCORE-5 keeps RUNNERCORE-4's structural payload contract and targets
    # the remaining self-reproduction policy/attach hot path without sharing
    # mutable child policy objects. Unsupported rows fail closed when requested.
    runnercore5_profile_enable: bool = False
    runnercore5_candidate_enable: bool = False
    runnercore5_native_enable: bool = False
    runnercore5_fail_closed_on_unsupported: bool = True
    runnercore5_allow_birth_boundary_agent_materialization: bool = False
    runnercore5_flat_self_policy_rng_enable: bool = False
    runnercore5_direct_newborn_mlp_bank_attach_enable: bool = False
    runnercore5_c1_canonical_boundary_fastpath_enable: bool = False
    runnercore5_write_summary: bool = True
    runnercore5_summary_filename: str = "runnercore5_child_construction_ownership_summary.json"

    # Optional RUNNERCORE-6 resident child-row lane. Default OFF.
    # RUNNERCORE-6 keeps deterministic child materialization for mirror
    # compatibility, but attaches newborn resident rows directly instead of
    # using the generic Agent mirror importer.
    runnercore6_profile_enable: bool = False
    runnercore6_candidate_enable: bool = False
    runnercore6_native_enable: bool = False
    runnercore6_fail_closed_on_unsupported: bool = True
    runnercore6_allow_birth_boundary_agent_materialization: bool = False
    runnercore6_resident_newborn_row_attach_enable: bool = False
    runnercore6_trusted_canonical_genome_materialization_enable: bool = False
    runnercore6_write_summary: bool = True
    runnercore6_summary_filename: str = "runnercore6_resident_child_row_summary.json"

    # Optional RUNNERCORE-7 compact child policy-row lane. Default OFF.
    # This is only valid for guarded no-mate self reproduction where the child
    # MLP policy row is byte-identical to the parent and the RNG budget is still
    # consumed through the RUNNERCORE-5 flat self-policy contract.
    runnercore7_self_mlp_policy_source_alias_enable: bool = False
    runnercore7_self_gru_policy_source_alias_enable: bool = False
    runnercore7_direct_newborn_gru_bank_attach_enable: bool = False
    runnercore7_static_reproduction_threshold_fastpath_enable: bool = False

    # Optional RUNNERCORE-9 C1 payload / lazy newborn lane. Default OFF.
    # The first guarded C1 slice skips only scalar-op normalization that is
    # topology-preserving. Lazy newborn materialization remains a separate gate
    # so compatibility exports can fail closed when unsupported.
    runnercore9_profile_enable: bool = False
    runnercore9_candidate_enable: bool = False
    runnercore9_native_enable: bool = False
    runnercore9_fail_closed_on_unsupported: bool = True
    runnercore9_c1_trace_enable: bool = False
    runnercore9_c1_scalar_op_normalize_skip_enable: bool = False
    runnercore9_lazy_newborn_materialization_enable: bool = False
    runnercore9_write_summary: bool = True
    runnercore9_summary_filename: str = "runnercore9_c1_payload_or_lazy_newborn_summary.json"

    # Optional RUNNERCORE-10 typed child genome / policy-row lane. Default OFF.
    # This keeps RUNNERCORE-7 policy-bank/Hmem row wins and defers compatibility
    # Agent materialization for resident newborn rows whose policy rows are
    # already directly attached from the typed child contract.
    runnercore10_profile_enable: bool = False
    runnercore10_candidate_enable: bool = False
    runnercore10_native_enable: bool = False
    runnercore10_fail_closed_on_unsupported: bool = True
    runnercore10_typed_child_contract_enable: bool = False
    runnercore10_defer_direct_attached_compat_enable: bool = False
    runnercore10_write_summary: bool = True
    runnercore10_summary_filename: str = "runnercore10_typed_child_genome_policy_row_summary.json"

    # Optional RUNNERCORE-11 C1 typed genome row ownership. Default OFF. This
    # projects canonical C1 output into a resident typed genome row and defers
    # compatibility Genome materialization until explicit export/debug boundaries.
    runnercore11_profile_enable: bool = False
    runnercore11_candidate_enable: bool = False
    runnercore11_native_enable: bool = False
    runnercore11_fail_closed_on_unsupported: bool = True
    runnercore11_c1_typed_rows_enable: bool = False
    runnercore11_fast_weighted_food_spawn_enable: bool = False
    runnercore11_generation_end_proxy_agents_enable: bool = False
    runnercore11_write_summary: bool = True
    runnercore11_summary_filename: str = "runnercore11_c1_typed_genome_row_summary.json"

    # Optional RUNNERCORE-12 C1 typed-row digest reuse. Default OFF. This keeps
    # the R11 typed-row semantics but avoids re-serializing the final C1 payload
    # for the row hash after the operator pipeline already owns that payload.
    runnercore12_c1_payload_digest_reuse_enable: bool = False

    # Optional RUNNERCORE-13 C1 projection digest sampling. Default OFF. The
    # payload hash remains complete, but the diagnostic projection digest is
    # only computed for retained trace samples instead of every typed row.
    runnercore13_c1_projection_digest_sample_only_enable: bool = False

    # Optional RUNNERCORE-14 native C1 typed scalar projection preflight.
    # Default OFF; normal imports and simulation do not require a compiler or
    # shared library. This is parity/ownership groundwork for a later live
    # native row builder, not an authoritative path by itself.
    runnercore14_native_c1_projection_enable: bool = False
    runnercore14_native_c1_projection_shadow_enable: bool = False
    runnercore14_native_c1_projection_fail_closed: bool = False
    runnercore14_native_c1_projection_library_path: str = ""

    # Optional RUNNERCORE-15 C1 no-op normalization skip. Default OFF. This
    # preserves C1 RNG/operator accounting but skips per-operator canonical
    # normalization when a catalog operator proves it made no structural change.
    runnercore15_c1_noop_normalize_skip_enable: bool = False

    # Optional RUNNERCORE-16 C1 empty-structure fast path. Default OFF. This
    # keeps C1 schedule/draw/operator accounting but directly builds the
    # canonical empty modules/connections payload instead of paying the generic
    # canonicalizer for known no-op structural children.
    runnercore16_c1_empty_structure_fastpath_enable: bool = False
    runnercore16_c1_empty_structure_fastpath_fail_closed: bool = False

    # Optional RUNNERCORE-17 resident birth-parent path. Default OFF. This
    # lets still-resident C1 typed child rows act as later birth parents without
    # forcing full compatibility Agent materialization at the birth boundary.
    runnercore17_resident_birth_parent_enable: bool = False
    runnercore17_resident_birth_parent_fail_closed: bool = False

    # Optional RUNNERCORE-18 resident-parent minimal refresh. Default OFF.
    # R17 refreshes full parent mirror stats for every resident birth candidate.
    # R18 narrows that to precheck fields first, then performs a full refresh
    # only for resident parents that reach child creation.
    runnercore18_resident_parent_minimal_refresh_enable: bool = False

    # Optional RUNNERCORE-19 resident maintenance arrays. Default OFF. This
    # owns the deterministic Phase A2 maintenance tick inside runnercore state,
    # avoiding per-tick Agent synchronization when maintenance is enabled.
    runnercore19_maintenance_arrays_enable: bool = False

    # Optional RUNNERCORE-20 resident policy-observation arrays. Default OFF.
    # Stores active sensor channel/radius rows and chemistry observation
    # diagnostics in runnercore state so policy inputs can match legacy W6
    # observation transforms without per-tick Agent synchronization.
    runnercore20_policy_observation_arrays_enable: bool = False

    # Optional RUNNERCORE-21 resident cost-charge scalar array. Default OFF.
    # Imports the accepted CostChargeSurface scalar into resident rows so env
    # apply can charge W6 cognitive/perception costs without per-tick phenotype
    # materialization.
    runnercore21_cost_charge_surface_arrays_enable: bool = False

    # Optional RUNNERCORE-22 full birth-boundary bridge. Default OFF. This keeps
    # legacy mate/patch/chemistry birth semantics available for W6 by exporting
    # all alive rows only at the birth event boundary, then importing changed
    # rows/newborns back into resident arrays. It is a compatibility bridge, not
    # a native/array-owned birth implementation.
    runnercore22_full_birth_boundary_bridge_enable: bool = False

    # Optional RUNNERCORE-23 jungle predator pressure bridge. Default OFF. This
    # exports alive rows before the legacy jungle pressure step and imports them
    # back after health/death mutations, preserving existing Patch/Jungle
    # semantics while keeping the rest of the tick resident.
    runnercore23_jungle_predator_bridge_enable: bool = False

    # Optional RUNNERCORE-24 native jungle predator arrays. Default OFF. This
    # applies Jungle predator pressure directly to resident health/alive arrays
    # while preserving legacy RNG draw order and event counters.
    runnercore24_native_jungle_predator_arrays_enable: bool = False

    # Optional RUNNERCORE-26 jungle predator vectorized sampler. Default OFF.
    # Requires RUNNERCORE-24 and batches the positive-risk RNG draws while
    # preserving sorted-agent draw order and native event semantics.
    runnercore26_vectorized_jungle_predator_sampling_enable: bool = False

    # Optional RUNNERCORE-27 full-birth cooldown prefilter. Default OFF.
    # Requires RUNNERCORE-22. Rows with repro_cooldown > 1 cannot reproduce and
    # consume no RNG this tick, so they can be decremented in resident arrays
    # and skipped at the legacy birth boundary when funnel event writing is off.
    runnercore27_full_birth_cooldown_prefilter_enable: bool = False

    # Optional RUNNERCORE-30 full-birth no-RNG branch prefilter. Default OFF.
    # Requires RUNNERCORE-22. Owns only pre-birth branches that consume no RNG
    # (self gates and isolated no-mate rows), preserving developmental side
    # effects before removing those rows from the legacy birth scan.
    runnercore30_full_birth_branch_prefilter_enable: bool = False

    # Optional RUNNERCORE-31 full-birth static developmental threshold fastpath.
    # Default OFF. Inside the legacy full-birth loop, no-module/default-program
    # genomes can use the genome reproduction_threshold directly while
    # synthesizing the same developmental cache and stage-counter side effects.
    runnercore31_full_birth_static_dev_threshold_enable: bool = False

    # Optional RUNNERCORE-32 full-birth inline grid scan. Default OFF. Keeps the
    # legacy mate/spawn decision order and RNG calls, but reads the resident env
    # grid arrays directly instead of routing thousands of birth-loop checks
    # through small method/Numba-wrapper calls.
    runnercore32_full_birth_inline_grid_scan_enable: bool = False

    # Optional RUNNERCORE-33 stride-1 direct action path. Default OFF. When the
    # policy cache stride is 1, cache hits are impossible, so this calls the same
    # sorted `_compute_actions_k(k=1)` refill producer without writing unusable
    # one-tick cache rows.
    runnercore33_stride1_direct_action_enable: bool = False

    # Optional RUNNERCORE-34 resident full-birth decision path. Default OFF.
    # This is a narrow mate-required W6 bridge: keep canonical child creation
    # and newborn bank sync, but decide parent/mate/spawn/failure branches from
    # resident arrays so successful parents are the only birth-decision mirrors
    # that need explicit refresh.
    runnercore34_full_birth_resident_decision_enable: bool = False

    # Optional RUNNERCORE-42 patch birth-context cache. Default OFF. Inside
    # RUNNERCORE-34's resident full-birth decision path, maintain the
    # Patch/Jungle diagnostic context maps across successful birth application
    # instead of rebuilding local density and derived patch arrays for every
    # sampled birth/emission context.
    runnercore42_full_birth_patch_context_cache_enable: bool = False

    # Optional RUNNERCORE-43 mate-birth threshold tick cache. Default OFF.
    # Inside RUNNERCORE-34's resident birth scan, cache the already-resolved
    # reproduction threshold for each agent during one tick. The first lookup
    # still performs the canonical developmental/static threshold side effects;
    # repeated mate-candidate checks reuse the same value.
    runnercore43_full_birth_threshold_tick_cache_enable: bool = False

    # Optional RUNNERCORE-47 developmental trace buffered writer. Default OFF.
    # Buffers developmental stage/lock trace JSONL lines in memory and flushes
    # them at the existing generation-end pending-trace barrier, preserving
    # line content and order while avoiding one open/write/close per trace.
    runnercore47_developmental_trace_buffered_emit_enable: bool = False
    # Developmental JSONL is diagnostic output and may be disabled by matched
    # production benchmark profiles without changing developmental semantics.
    developmental_trace_write_enable: bool = True

    # Optional RUNNERCORE-45 jungle predator patch-context cache. Default OFF.
    # Reuses one Patch/Jungle context map cache inside the RUNNERCORE-24
    # predator loop, updating cached density after predator deaths so later
    # same-tick diagnostic contexts see the same occupancy as legacy sampling.
    runnercore45_jungle_predator_context_cache_enable: bool = False

    # Optional RUNNERCORE-48 resident chemistry observation vectorization.
    # Default OFF. Inside RUNNERCORE-20's policy-observation array path, compute
    # the deterministic chemistry fidelity/bias transform for all requested
    # policy rows from resident chemistry arrays instead of one scalar helper
    # call per agent.
    runnercore48_vectorized_chemistry_observation_enable: bool = False

    # Optional RUNNERCORE-49 mate-birth chemistry viability tick cache. Default
    # OFF. Inside RUNNERCORE-34's resident mate-birth scan, cache the
    # deterministic chemistry reproduction viability result per agent during one
    # tick so repeated mate-candidate checks do not reclamp the same chemistry
    # row. The first lookup still updates the canonical Agent stat mirror.
    runnercore49_repro_viability_tick_cache_enable: bool = False

    # Optional RUNNERCORE-50 mate-birth spawn selection simplification. Default
    # OFF. Inside RUNNERCORE-34's resident mate-birth scan, keep the exact
    # canonical RNG shuffle of the eight parent/mate neighbor cells, but select
    # the first valid shuffled spawn cell directly instead of building and
    # de-duplicating an intermediate candidate list that is only used for its
    # first element.
    runnercore50_spawn_first_valid_enable: bool = False

    # Optional RUNNERCORE-25 full-birth resident newborn row bridge. Default
    # OFF. When RUNNERCORE-22 delegates W6 mate/patch/chemistry birth decisions
    # to legacy reproduction, this keeps those legacy decisions but asks
    # Population for a resident child-row spec instead of immediately
    # materializing a compatibility Agent.
    runnercore25_full_birth_resident_newborn_attach_enable: bool = False

    # Optional CPU-only policy forward fast path. Default OFF. When enabled on a
    # CPU policy/bank run, execute packed MLP/GRU policy rows with NumPy instead
    # of staging through torch tensors. CUDA and non-CPU banks stay on the
    # existing torch path.
    perf_policy_cpu_numpy_forward_enable: bool = False

    # Optional env/action phenotype and developmental-expression hotpath
    # diagnostics. Default OFF; writes compact metrics-only summaries.
    env_hotpath_expression_diagnostics_enable: bool = False
    env_hotpath_expression_diagnostics_schema_version: int = 1
    env_hotpath_expression_diagnostics_write_events: bool = False

    # Optional env-apply decomposition and canonicalization audit diagnostics.
    # Default OFF; writes compact metrics-only summaries.
    env_hotpath_apply_decomposition_enable: bool = False
    env_hotpath_apply_decomposition_schema_version: int = 1

    # Optional CXXCORE-1 env-apply shadow parity/timing probe. Default OFF.
    # Python remains authoritative; the shadow path only reads copied state.
    cxxcore_env_apply_shadow_enable: bool = False
    cxxcore_env_apply_shadow_schema_version: int = 1
    cxxcore_env_apply_shadow_backend: str = "python"
    cxxcore_env_apply_shadow_compare_enable: bool = True
    cxxcore_env_apply_shadow_fail_fast: bool = False
    cxxcore_env_apply_shadow_write_summary: bool = True
    cxxcore_env_apply_shadow_summary_filename: str = "cxxcore_env_apply_shadow_summary.json"

    # Optional phenotype path decomposition diagnostics. Default OFF; writes a
    # compact metrics-only summary for Python-side cost attribution.
    env_hotpath_phenotype_decomposition_enable: bool = False
    env_hotpath_phenotype_decomposition_schema_version: int = 1

    # Optional ENVHOT-7 developmental expression residual diagnostics. Default
    # OFF; writes compact call/count/timing summaries without changing behavior.
    env_hotpath_developmental_residual_diagnostics_enable: bool = False
    env_hotpath_developmental_residual_diagnostics_schema_version: int = 1

    # Optional ENVHOT-8 lock-aware runner/phenotype developmental-expression
    # equality diagnostics. Default OFF; compares without reusing values.
    env_hotpath_developmental_cache_equality_diagnostics_enable: bool = False
    env_hotpath_developmental_cache_equality_schema_version: int = 1

    # Optional ENVHOT-9 effective-projection contract diagnostics. Default OFF;
    # compares projection-only values without changing phenotype behavior.
    env_hotpath_developmental_effective_projection_diagnostics_enable: bool = False
    env_hotpath_developmental_effective_projection_schema_version: int = 1

    # Optional ENVHOT-4 performance shortcut. Default OFF; when enabled,
    # express_cognitive_phenotype may use fresh factory defaults for proven
    # plain-disabled inactive helper paths only.
    env_hotpath_inactive_phenotype_defaults_enable: bool = False

    # Optional ENVHOT-5 performance shortcut. Default OFF; when enabled, the
    # env-apply cost-charging path computes the charged scalar cost directly
    # instead of assembling a full budget ledger that is immediately discarded.
    env_hotpath_budget_ledger_scalar_fastpath_enable: bool = False

    # -----------------------------
    # Inactivity / rest charge
    # -----------------------------
    rest_streak_free: int = 8
    rest_charge_energy: float = 0.25
    rest_charge_growth: float = 0.10
    rest_deficit_to_health: float = 1.0
    enable_rest_charge: bool = True

    # -----------------------------
    # Phase A2: delayed maintenance process
    # -----------------------------
    maintenance_enable: bool = True
    maintenance_delay_ticks: int = 3
    maintenance_base_effect: float = 0.10
    maintenance_diminishing_factor: float = 0.60
    maintenance_recent_decay: float = 0.92
    maintenance_energy_cost: float = 0.08
    maintenance_opportunity_cost_coeff: float = 0.35
    maintenance_domination_threshold: float = 0.75
    maintenance_domination_min_ticks: int = 8
    resource_maintenance_energy_cost_per_tick: float = 0.0
    initial_spatial_placement_profile: str = "heritage_random_v1"

    # -----------------------------
    # Phase A3: chemistry -> functional coupling
    # -----------------------------
    chemistry_functional_coupling_enable: bool = True
    # Action-cost chemistry is behavior-changing and expensive on the legacy
    # object loop, so existing broad chemistry coupling does not imply it.
    chemistry_action_cost_coupling_enable: bool = False
    chemistry_action_cost_min_mult: float = 1.0
    chemistry_action_cost_max_mult: float = 1.35
    chemistry_action_cost_strength: float = 0.9
    chemistry_action_cost_power: float = 1.0

    chemistry_observation_fidelity_min: float = 0.70
    chemistry_observation_fidelity_max: float = 1.00
    chemistry_observation_noise_abs_max: float = 0.15
    chemistry_observation_fidelity_strength: float = 0.9
    chemistry_observation_noise_strength: float = 0.8

    chemistry_repro_viability_enable: bool = True
    chemistry_repro_viability_floor: float = 0.20
    chemistry_repro_viability_strength: float = 0.80
    chemistry_repro_chronic_block_threshold: float = 0.95

    # Phase A chemistry observability thresholds (metrics.csv burden fractions).
    chem_acute_high_threshold: float = 0.70
    chem_chronic_high_threshold: float = 0.70

    selection_forbid_chemistry_scalar: bool = True

    a3_recovery_taxonomy_enable: bool = True
    a3_recovery_baseline_ema_alpha: float = 0.15
    a3_recovery_return_fitness_ratio: float = 0.92
    a3_recovery_return_drift_ratio_max: float = 1.10
    a3_recovery_shift_fitness_ratio_min: float = 0.70

    # -----------------------------
    # Phase B2: module interfaces + connection budgets (campaign-frozen)
    # -----------------------------
    # NOTE: These are governance-declared mirrors for run-config auditing.
    # Runner startup enforces them to frozen constants (strict mismatch -> fail-fast).
    # They are not free-form runtime tuning knobs.
    genome_b2_interfaces_enable: bool = True
    genome_b2_invalid_policy: str = "clip"  # strict mode always upgrades to reject.
    genome_b2_max_modules_per_agent: int = 12
    genome_b2_max_connections_per_agent: int = 32
    genome_b2_max_module_fanin: int = 6
    genome_b2_max_module_fanout: int = 6

    # -----------------------------
    # Phase B3: locality constraints (campaign-frozen governance mirrors)
    # -----------------------------
    genome_b3_locality_enable: bool = True
    genome_b3_max_connection_distance: float = 0.65
    genome_b3_distance_cost_scale: float = 1.0
    genome_b3_distance_cost_max: float = 1.0

    # -----------------------------
    # Phase C1: deterministic structural operator framework (campaign-frozen mirrors)
    # -----------------------------
    genome_c1_enable: bool = True
    genome_c1_catalog_version: str = "c1_operator_catalog_v2_structural_bootstrap"
    genome_c1_max_ops_per_offspring: int = 2
    genome_c1_invalid_operator_policy: str = "reject"
    structural_bootstrap_enable: bool = False
    structural_identity_schema_version: str = "structural_identity_v1"
    structural_ancestry_schema_version: str = "structural_ancestry_v1"
    topology_position_schema_version: str = "topology_position_v1"

    # -----------------------------
    # Phase C2: deterministic lineage-aware recombination (campaign-frozen mirrors)
    # -----------------------------
    genome_c2_enable: bool = True
    genome_c2_recombination_version: str = "c2_deterministic_homology_recombination_v2"
    genome_c2_alignment_key: str = "homology_id"
    genome_c2_unmatched_policy: str = "keep_all_sorted"
    genome_c2_legacy_unresolved_policy: str = "canonical_parent_set_no_cross_alignment"

    # REP-1C: default-off structural payload activation for controlled audits.
    structural_payload_activation_enable: bool = False
    structural_payload_seed_profile: str = "none"
    structural_payload_seed_schema_version: str = "structural_payload_seed_v1"
    structural_payload_seed_ablation: str = "none"

    # REP-2B: default-off live structural semantics for controlled audits.
    structural_live_semantics_enable: bool = False
    structural_live_semantics_profile: str = "none"
    structural_live_semantics_schema_version: str = "structural_live_semantics_v1"
    structural_live_effect_scale: float = 1.0
    # ESCAP-3 bounded CPU/reference module executor. Independent and default-off
    # so REP-2B scalar compilation retains byte-for-byte disabled behavior.
    structural_live_modules_enable: bool = False
    structural_live_modules_profile: str = "none"
    structural_live_modules_schema_version: str = "escap3_live_module_runtime_v1"
    # ESCAP-4 bounded module-local routing of existing scalar observation
    # meanings.  The separate mutation attempt budget leaves every pre-ESCAP-4
    # C1 scheduler profile byte-for-byte unchanged while this gate is off.
    structural_observation_routing_enable: bool = False
    structural_observation_routing_profile: str = "none"
    structural_observation_routing_schema_version: str = "escap4_observation_routing_v1"
    structural_observation_routing_registry_version: str = "perception_routing_contract_v1"
    structural_observation_routing_max_bindings_per_module: int = 2
    structural_observation_routing_max_bindings_per_agent: int = 8
    structural_observation_routing_operator_profile: str = "escap4_routing_binding_operator_v1"
    structural_observation_routing_operator_attempts: int = 1
    # ESCAP-5 bounded module-output routing over existing action-logit
    # meanings.  The explicit operator lane remains absent from every legacy
    # C1 schedule while this independent feature gate is off.
    structural_output_routing_enable: bool = False
    structural_output_routing_profile: str = "none"
    structural_output_routing_schema_version: str = "escap5_output_routing_v1"
    structural_output_routing_registry_version: str = "eiel_action_semantic_routing_registry_v1"
    structural_output_routing_max_bindings_per_module: int = 2
    structural_output_routing_max_bindings_per_agent: int = 8
    structural_output_routing_operator_profile: str = "escap5_output_binding_operator_v1"
    structural_output_routing_operator_attempts: int = 1
    structural_output_routing_max_binding_gain: float = 1.0
    structural_output_routing_max_action_residual: float = 0.25
    structural_output_routing_arbitration_version: str = "escap5_output_arbitration_v1"
    structural_output_routing_authority_order_version: str = "escap5_before_existing_authority_v1"
    structural_output_routing_combined_rest_compatibility_version: str = "escap5_combined_rest_residual_v1"
    # ESCAP-6 bounded developmental expression over existing structural
    # modules.  The gate is independent and default OFF.  It adds no age
    # counter, tensor dimension, observation/action meaning, or runtime lane;
    # the fixed values below are the strict three-stage compatibility surface.
    structural_development_enable: bool = False
    structural_development_profile: str = "none"
    structural_development_module_schema_version: str = DEVELOPMENTAL_MODULE_SCHEMA_VERSION
    structural_development_stage_profile_version: str = DEVELOPMENTAL_STAGE_PROFILE_VERSION
    structural_development_stage_ids: tuple[str, ...] = DEVELOPMENTAL_STAGE_IDS
    structural_development_mature_start_step: int = MATURE_START_STEP
    structural_development_late_start_step: int = LATE_START_STEP
    structural_development_min_stage_gain: float = MIN_STAGE_GAIN
    structural_development_max_stage_gain: float = MAX_STAGE_GAIN
    structural_development_dependency_identity_authority: str = DEPENDENCY_IDENTITY_AUTHORITY
    structural_development_max_prerequisites_per_module: int = MAX_PREREQUISITES_PER_MODULE
    structural_development_max_dependency_edges_per_agent: int = MAX_DEPENDENCY_EDGES_PER_AGENT
    structural_development_cycle_policy: str = CYCLE_POLICY
    structural_development_inactive_state_policy: str = INACTIVE_STATE_POLICY
    structural_development_inactive_output_policy: str = INACTIVE_OUTPUT_POLICY
    structural_development_latch_version: str = LATCH_VERSION
    structural_development_latch_semantics: str = LATCH_SEMANTICS
    structural_development_operator_profile: str = DEVELOPMENT_OPERATOR_PROFILE
    structural_development_operator_attempts: int = DEVELOPMENT_OPERATOR_ATTEMPTS
    structural_development_operator_draws_per_attempt: int = DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT
    structural_development_migration_rng_draws: int = MIGRATION_RNG_DRAWS
    structural_development_supported_lane: str = SUPPORTED_LANE
    # ESCAP-7 richer recurrence remains inside the existing fixed one-scalar
    # module-state allocation.  This independent gate is default OFF, so an
    # ESCAP-6 configuration neither admits the new operator nor gains new
    # checkpoint metadata or lane restrictions.
    structural_gated_recurrent_enable: bool = False
    structural_gated_recurrent_profile: str = "none"
    structural_gated_recurrent_schema_version: str = GATED_RUNTIME_SCHEMA_VERSION
    structural_gated_recurrent_operator: str = GATED_OPERATOR
    structural_gated_recurrent_parameter_schema_version: str = GATED_RECORD_SCHEMA_VERSION
    structural_gated_recurrent_equation_version: str = GATED_EQUATION_VERSION
    structural_gated_recurrent_numeric_policy_version: str = GATED_NUMERIC_VERSION
    structural_gated_recurrent_state_width: int = GATED_STATE_WIDTH
    structural_gated_recurrent_neutral_eligibility_version: str = (
        GATED_NEUTRAL_ELIGIBILITY_VERSION
    )
    structural_gated_recurrent_neutral_latent_state_policy: str = (
        GATED_LATENT_STATE_POLICY_VERSION
    )
    structural_gated_recurrent_mutation_operator_profile: str = (
        GATED_MUTATION_PROFILE
    )
    structural_gated_recurrent_operator_attempts: int = 1
    structural_gated_recurrent_operator_draws_per_attempt: int = 2
    structural_gated_recurrent_draw_bin_count: int = 5
    structural_gated_recurrent_decomposition_version: str = (
        GATED_DRAW_DECOMPOSITION_VERSION
    )
    structural_gated_recurrent_mutation_budget_version: str = (
        GATED_MUTATION_BUDGET_VERSION
    )
    structural_gated_recurrent_operation_budget_version: str = (
        GATED_OPERATION_BUDGET_VERSION
    )
    structural_gated_recurrent_duplication_policy_version: str = (
        GATED_DUPLICATION_POLICY_VERSION
    )
    structural_gated_recurrent_promoter_policy_version: str = (
        GATED_PROMOTER_POLICY_VERSION
    )
    structural_gated_recurrent_fusion_policy_version: str = (
        GATED_FUSION_POLICY_VERSION
    )
    structural_gated_recurrent_c2_compatibility_version: str = (
        GATED_C2_COMPATIBILITY_VERSION
    )
    structural_gated_recurrent_supported_lane: str = (
        GATED_SUPPORTED_LANE
    )
    structural_gated_recurrent_migration_rng_draws: int = 0
    # ESCAP-8 makes fixed-envelope modular growth mechanically sustainable.
    # The independent gate is default OFF, so ESCAP-7 schedules, genome bytes,
    # checkpoint identity material, and lane validation remain exact.
    structural_sustainable_growth_enable: bool = False
    structural_sustainable_growth_profile: str = "none"
    structural_sustainable_growth_checkpoint_schema_version: str = (
        SUSTAINABLE_GROWTH_CHECKPOINT_SCHEMA_VERSION
    )
    structural_sustainable_growth_activity_schema_version: str = ACTIVITY_SCHEMA_VERSION
    structural_sustainable_growth_duplication_neutralization_version: str = (
        DUPLICATION_NEUTRALIZATION_VERSION
    )
    structural_sustainable_growth_refinement_operator_version: str = (
        REFINEMENT_OPERATOR_VERSION
    )
    structural_sustainable_growth_specialization_telemetry_version: str = (
        SPECIALIZATION_TELEMETRY_VERSION
    )
    structural_sustainable_growth_prune_eligibility_version: str = PRUNE_ELIGIBILITY_VERSION
    structural_sustainable_growth_prune_cleanup_version: str = PRUNE_CLEANUP_VERSION
    structural_sustainable_growth_repack_version: str = REPACK_VERSION
    structural_sustainable_growth_fusion_policy_version: str = (
        SUSTAINABLE_GROWTH_FUSION_POLICY_VERSION
    )
    structural_sustainable_growth_cost_schema_version: str = COST_SCHEMA_VERSION
    structural_sustainable_growth_cost_profile: str = COST_PROFILE
    structural_sustainable_growth_cost_mode: str = COST_MODE
    structural_sustainable_growth_cost_coefficients: dict[str, float] = field(
        default_factory=lambda: {
            str(key): float(value) for key, value in COST_COEFFICIENTS.items()
        }
    )
    structural_sustainable_growth_bloat_classification_version: str = (
        BLOAT_CLASSIFICATION_VERSION
    )
    structural_sustainable_growth_operator_profile: str = GROWTH_MUTATION_PROFILE
    structural_sustainable_growth_operator_attempts: int = GROWTH_MUTATION_ATTEMPTS
    structural_sustainable_growth_operator_draws_per_attempt: int = GROWTH_MUTATION_DRAWS
    structural_sustainable_growth_c1_lane_map: tuple = tuple(GROWTH_MUTATION_BINS)
    structural_sustainable_growth_c2_compatibility_version: str = (
        SUSTAINABLE_GROWTH_C2_COMPATIBILITY_VERSION
    )
    structural_sustainable_growth_supported_lane: str = (
        SUSTAINABLE_GROWTH_SUPPORTED_LANE
    )
    structural_sustainable_growth_migration_rng_draws: int = (
        SUSTAINABLE_GROWTH_MIGRATION_RNG_DRAWS
    )
    structural_action_opportunity_reporting_enable: bool = False
    structural_action_margin_reporting_enable: bool = False
    structural_counterfactual_action_response_reporting_enable: bool = False
    structural_decision_margin_reporting_enable: bool = False
    structural_decision_margin_event_sample_limit: int = 512
    structural_policy_action_family_mapping_reporting_enable: bool = False
    structural_action_consequence_telemetry_reporting_enable: bool = False
    policy_family_head_shadow_enable: bool = False
    policy_family_head_shadow_record_events: bool = False
    policy_family_head_residual_genome_enable: bool = False
    policy_family_head_residual_mutation_enable: bool = False
    policy_family_head_residual_shadow_enable: bool = False
    policy_family_head_residual_reporting_enable: bool = False
    policy_family_head_residual_component_bound: float = 1.0
    policy_family_head_residual_total_l1_cap: float = 2.0
    policy_family_head_residual_mutation_step: float = 0.1
    policy_family_head_residual_authority_enable: bool = False
    policy_family_head_residual_cost_enable: bool = False
    policy_family_head_residual_cost_coefficient: float = 0.0
    policy_family_head_binary_opportunity_gate_enable: bool = False
    policy_family_head_binary_opportunity_gate_reporting_enable: bool = False
    policy_family_head_binary_opportunity_gate_event_sample_limit: int = 2048
    policy_family_head_residual_authority_reporting_enable: bool = False
    policy_family_head_authority_choice_margin_reporting_enable: bool = False
    policy_family_head_authority_choice_margin_event_sample_limit: int = 2048
    policy_family_head_authority_magnitude_surface_reporting_enable: bool = False
    policy_family_head_authority_magnitude_surface_event_sample_limit: int = 2048
    policy_family_head_opportunity_conditioned_authority_candidate_reporting_enable: bool = False
    policy_family_head_opportunity_conditioned_authority_candidate_event_sample_limit: int = 2048
    policy_family_head_residual_fixed_arm: str = ""
    policy_architecture_context: str = ""
    structural_selection_funnel_reporting_enable: bool = False
    structural_tick_reproduction_funnel_reporting_enable: bool = False
    structural_variant_identity_reporting_enable: bool = False
    structural_variant_event_join_reporting_enable: bool = False
    # REP-RH0: default-off dormant A/B resource-handling representation.
    # Enabling this mutates only the inherited genome field; no live resource,
    # action, reward, survival, or policy behavior consumes the value here.
    resource_handling_representation_enable: bool = False
    resource_handling_representation_reporting_enable: bool = False

    # -----------------------------
    # Environmental shock schedule
    # -----------------------------
    shock_period: int = 10
    shock_duration: int = 3
    shock_food_spawn_mult: float = 0.50
    shock_action_cost_mult: float = 1.15
    shock_injury_recover_mult: float = 0.50

    shock_start_prob: float = 0.03
    shock_duration_min: int = 2
    shock_duration_max: int = 4
    shock_cooldown_min: int = 20
    shock_cooldown_max: int = 45

    # -----------------------------
    # Sprint B: genome diversity + collapse detection
    # -----------------------------
    diversity_collapse_window: int = 20
    diversity_collapse_threshold: float = 0.010
    diversity_collapse_streak_required: int = 10

    # -----------------------------
    # Theme 5: biological lineage observability (diagnostic only)
    # -----------------------------
    lineage_observability_enable: bool = True
    lineage_dominance_threshold: float = 0.80
    lineage_dominance_streak_required: int = 5
    lineage_min_active_lineages: int = 2
    lineage_entropy_minimum: float = 0.10
    lineage_reproduction_dominance_threshold: float = 0.80
    lineage_reproduction_dominance_streak_required: int = 5
    lineage_qd_archive_enable: bool = True
    lineage_qd_archive_top_n: int = 8
    lineage_qd_archive_min_score: float = 0.25
    lineage_qd_archive_cooldown_generations: int = 0
    lineage_collapse_forensics_enable: bool = False
    lineage_collapse_forensics_schema_version: int = 1
    lineage_collapse_effective_threshold: float = 1.5
    lineage_collapse_top1_share_threshold: float = 0.95
    lineage_collapse_minority_share_threshold: float = 0.05
    lineage_collapse_patch_context_top_n: int = 8
    lineage_collapse_json_top_n: int = 8
    birth_skew_mechanism_reporting_enable: bool = False
    birth_skew_mechanism_schema_version: int = 1
    birth_skew_mechanism_top_n: int = 3

    # Theme 5 diagnostic: lineage reproduction funnel reporting.
    # Reporting-only and OFF by default. This must not alter selection,
    # reproduction, cull/refill, ecology, rewards, archive entries, or founder
    # initialization defaults.
    lineage_reproduction_funnel_reporting_enable: bool = False
    lineage_reproduction_funnel_schema_version: int = 1
    lineage_reproduction_funnel_write_per_generation: bool = True
    lineage_reproduction_funnel_write_events: bool = False
    lineage_reproduction_funnel_event_schema_version: int = 3
    lineage_reproduction_funnel_write_scan_action_events: bool = False
    lineage_reproduction_funnel_event_output_profile: str = "full"
    lineage_reproduction_funnel_incremental_summary_cache_enable: bool = False
    lineage_reproduction_funnel_include_patch_context: bool = True
    lineage_reproduction_funnel_include_basin_context: bool = True
    mutation_weather_enabled: bool = False
    mutation_weather_profile: str = ""
    mutation_weather_seed_salt: str = ""
    mutation_weather_max_scalar_multiplier: float = 10.0
    mutation_weather_apply_to_scalar_sigma: bool = True
    mutation_weather_apply_to_mutation_probability: bool = False
    mutation_weather_structural_enabled: bool = False
    mutation_weather_fixture_generations: str = ""
    mutation_weather_fixed_scalar_multiplier: float = 1.0
    path2_roz_reporting_enable: bool = False
    path2_roz_schema_version: int = 1
    path2_roz_zone_mode: str = "static_center_manhattan"
    path2_roz_window_period_steps: int = 40
    path2_roz_window_duration_steps: int = 8
    path2_roz_zone_radius: int = 0
    path2_roz_schedule_mode: str = "static_periodic_compat"
    path2_roz_schedule_id: str = ""
    path2_roz_schedule_phase_offset_steps: int = 0
    path2_roz_schedule_zones_json: str = ""
    lro1_reproductive_opportunity_enable: bool = False
    lro1_reproductive_opportunity_mode: str = "local_patch_carrying_capacity_v1"
    lro1_reproductive_opportunity_apply_stage: str = "pre_attempt_try_prob"
    lro1_reproductive_opportunity_min_multiplier: float = 0.35
    lro1_reproductive_opportunity_max_multiplier: float = 1.0
    lro1_reproductive_opportunity_weights_json: str = (
        '{"density":-0.35,"waste":-0.25,"scent":-0.10,"shelter_cover":0.20,"food":0.10,"hazard":-0.20}'
    )
    lro1_reproductive_opportunity_report_fields_enable: bool = True
    lro4_offspring_dispersal_enable: bool = False
    lro4_offspring_dispersal_schema_version: int = 1
    lro4_offspring_dispersal_mode: str = "crowding_conditioned_radius_v1"
    lro4_offspring_dispersal_base_radius: int = 1
    lro4_offspring_dispersal_max_radius: int = 3
    lro4_offspring_dispersal_density_threshold: float = 4.0
    lro4_offspring_dispersal_waste_threshold: float = 0.25
    lro4_offspring_dispersal_no_space_trigger_enable: bool = True
    lro4_offspring_dispersal_deterministic_ring_order: bool = True
    lro4_offspring_dispersal_report_fields_enable: bool = True

    # BRANCH-2 compact branch observability. Reporting-only and OFF by default.
    # When disabled it must create no files, draw no RNG, and alter no trajectory.
    branch_observability_enable: bool = False
    branch_observability_birth_edges_enable: bool = True
    branch_observability_lifecycle_enable: bool = True
    branch_observability_descriptor_snapshot_enable: bool = True
    branch_observability_output_profile: str = "compact_v1"
    branch_observability_run_id: str = ""
    branch_observability_warning_row_threshold: int = 100000
    branch_niche_observability_enable: bool = False
    branch_niche_lifecycle_enable: bool = True
    branch_niche_strategy_compression_enable: bool = True
    branch_niche_output_profile: str = "compact_v1"
    branch_niche_warning_row_threshold: int = 100000

    # Phase II-B1: initial founder/lineage balance (initialization-only).
    # OFF by default to preserve legacy one-founder-per-agent initialization.
    lineage_founder_balance_enable: bool = False
    lineage_founder_group_count: int = 8
    lineage_founder_min_agents_per_group: int = 4
    lineage_founder_assignment_mode: str = "deterministic_round_robin"
    lineage_founder_balance_schema_version: int = 1

    # Phase IV lineage evidence substrate: initialization-only founder spatial
    # separation diagnostic. OFF by default and never applies post-init lineage
    # protection, quotas, rewards, rescue, reseeding, or reproduction changes.
    lineage_founder_spatial_separation_enable: bool = False
    lineage_founder_spatial_group_count: int = 0
    lineage_founder_spatial_min_distance: int = 0
    lineage_founder_spatial_assignment_mode: str = "none"
    lineage_founder_spatial_use_patch_basins: bool = False
    lineage_founder_spatial_schema_version: int = 1

    # -----------------------------
    # Early-life weighted fitness (existing feature)
    # -----------------------------
    early_life_steps: int = 0
    early_life_weight: float = 1.0

    # -----------------------------
    # Episode-0 coherence selection pressure
    # -----------------------------

    # (A) Legacy combined gate: kept for backward compatibility.
    episode0_fitness_weight: float = 0.0

    # (B) Split knobs (preferred for causal tests).
    # If any of these is non-zero, selection uses split mode and ignores episode0_fitness_weight.
    episode0_pred_err_fitness_weight: float = 0.0
    episode0_thrashing_fitness_weight: float = 0.0
    episode0_drive_jitter_fitness_weight: float = 0.0
    episode0_selfharm_fitness_weight: float = 0.0

    # Per-signal weights INSIDE legacy combined mode (unitless).
    # These do NOT affect split mode (split mode already isolates each component).
    episode0_pred_err_weight: float = 1.0
    episode0_drive_jitter_weight: float = 1.0
    episode0_thrashing_weight: float = 1.0
    episode0_selfharm_weight: float = 3.0

    # Normalization caps for episode-0 signals (used in both modes).
    episode0_pred_err_cap: float = 5.0
    episode0_drive_jitter_cap: float = 5.0
    episode0_thrashing_rate_cap: float = 1.0
    episode0_selfharm_rate_cap: float = 0.25

    # -----------------------------
    # Step 3: developmental coupling logging (analysis only)
    # -----------------------------
    # Default OFF so it can't surprise you or break old workflows.
    devcoupling_enable: bool = False

    # Cohort split (quantiles) by episode-0 prediction error.
    # low = bottom q; high = top q
    devcoupling_low_q: float = 0.20
    devcoupling_high_q: float = 0.80

    # Minimum agents required to write a row for meaningful stats.
    devcoupling_min_agents: int = 32

    # ---------------------------------------------------------------------
    # Option C: Late-Life Predictive Coherence Pressure (analysis-clean)
    # ---------------------------------------------------------------------
    # If enabled, we apply a small scalar penalty to the *selection score* used
    # to choose parents, based on each agent's mean late-life prediction error.
    #
    # IMPORTANT:
    # - "Late-life" begins strictly AFTER the agent's first successful reproduction
    #   (stats.first_repro_step). This guarantees we do NOT reintroduce early-life
    #   pressure that would distort your confirmed early-repro dominance.
    # - The penalty is applied ONLY at selection time (runner_selection.py).
    # - No world mechanics change (no energy/health/resource coupling).
    #
    # Applied at parent selection time:
    #   late_mean = late_pred_err_sum / late_pred_err_count
    #   fitness_sel = max(0, fitness_base - late_pred_cost_weight * late_mean)
    #
    # Master enable (default OFF). Campaigns that study this pressure must
    # opt in explicitly so memory/perception assays do not inherit it.
    late_pred_cost_enable: bool = False
    # Keep tiny (order(s) of magnitude smaller than early-life weights).
    late_pred_cost_weight: float = 0.01

    # ---------------------------------------------------------------------
    # Life-history tradeoff: Late-life coherence efficiency bonus (minimal)
    # ---------------------------------------------------------------------
    # If enabled, AFTER first successful reproduction, agents with lower
    # late-life prediction error pay slightly less per-action energy cost.
    # This creates a real life-history tradeoff without changing learning,
    # adding new mechanics, or coupling to episode-0 fitness.
    #
    # Implementation: a per-tick cost multiplier passed into env.step_agents_batch:
    #   coherence = 1 - clamp01(late_pred_err_mean / late_coherence_eff_scale)
    #   cost_mult = 1 - late_coherence_eff_weight * coherence
    #
    # Applied only if stats.first_repro_step is set and survival_steps > first_repro_step.
    # Default OFF. Campaigns that study this pressure must opt in explicitly.
    late_coherence_eff_enable: bool = False
    # Strength of the efficiency bonus. Keep tiny (e.g., 0.01–0.05).
    late_coherence_eff_weight: float = 0.02
    # Normalization scale for late-life pred error mean (higher => weaker effect).
    late_coherence_eff_scale: float = 5.0
    # Require at least this many post-repro pred error samples before granting bonus.
    late_coherence_eff_min_count: int = 5
    # Floor on cost multiplier for safety (prevents extreme discounts).
    late_coherence_eff_floor: float = 0.50

    # ---------------------------------------------------------------------
    # Deprecated legacy: per-tick late-life energy tax (world-dynamics confound)
    # ---------------------------------------------------------------------
    # This older mechanic directly drains energy during life based on later-life
    # prediction error. It is NOT Option C and is kept only for backwards
    # experiments. OFF by default.
    late_pred_energy_tax_enable: bool = False
    late_pred_energy_tax_weight: float = 0.0
    late_pred_energy_tax_cap: float = 5.0
    late_pred_energy_tax_start_step: int = 20


    # -----------------------------
    # Adult-coherence capability coupling (Part B)
    # -----------------------------
    # If enabled, after adult_move_penalty_start_step (life steps since birth),
    # we compute an exponential moving average (EMA) of prediction error for each
    # agent (based on the same per-step pred_err used for metrics).
    #
    # We then apply a deterministic movement cost multiplier:
    #
    #   coherence = 1 - clamp01(adult_pred_err_ema / adult_move_penalty_scale)
    #   move_mult = 1 + adult_move_penalty_strength * (1 - coherence)
    #   move_mult = min(move_mult, adult_move_penalty_cap)
    #
    # This couples adult predictive coherence to *capability* (movement efficiency)
    # without changing episode-0 selection or adding new learning systems.

    # Master enable (default OFF).
    adult_move_penalty_enable: bool = False

    # When to start applying this adult coupling (life steps since birth).
    # Default matches late_pred_cost_start_step (post-episode-0).
    adult_move_penalty_start_step: int = 20

    # EMA update rate for adult prediction error. Higher = reacts faster.
    # Typical: 0.05–0.2.
    adult_pred_err_ema_alpha: float = 0.1

    # Normalization scale for pred_err_ema. Higher => weaker penalty.
    adult_move_penalty_scale: float = 5.0

    # Strength of the movement multiplier at full incoherence (before cap).
    # Example: 0.25 means up to +25% move cost at saturation.
    adult_move_penalty_strength: float = 0.0

    # Hard cap on the movement multiplier (safety guard).
    adult_move_penalty_cap: float = 3.0

    # -----------------------------
    # Temporal consistency pressure (Phase 4): Identity drift capability tax
    # -----------------------------
    # Goal: apply selection pressure against agents whose *internal organization*
    # drifts across their lifetime, WITHOUT adding memory/learning/new modules.
    #
    # Mechanism:
    # - Capture a fixed "probe signature" once at the end of episode-0:
    #     baseline_probs = softmax(policy_logits(probe_obs))
    # - In adulthood, periodically recompute probe_probs using the *current* internal
    #   state (for GRU agents this depends on hidden state).
    # - Drift is the L1 distance between current and baseline probe probabilities.
    # - Drift increases a movement cost multiplier (capability coupling), gated by age.
    #
    # Properties:
    # - Deterministic
    # - Local (per-agent)
    # - Configurable + falsifiable
    # - OFF by default

    identity_drift_enable: bool = False

    # Probe cadence (in ticks). Higher = cheaper.
    identity_probe_every_k: int = 8

    # When to start applying the drift penalty (life steps since birth).
    # Default aligns with adult_move_penalty_start_step if present.
    identity_drift_start_step: int = 20

    # Normalization scale for drift (L1 distance). Higher => weaker penalty.
    identity_drift_scale: float = 0.10

    # Strength of movement multiplier at full drift (before cap).
    identity_drift_strength: float = 0.0

    # Hard cap on the movement multiplier contributed by drift (safety guard).
    identity_drift_cap: float = 3.0

    # EMA alpha for drift smoothing (0 disables smoothing, 1 means no smoothing).
    identity_drift_ema_alpha: float = 0.25




    # -----------------------------
    # Checkpointing (Checkpoint v1)
    # -----------------------------
    # Save a checkpoint every K generations (0 disables).
    checkpoint_period_gen: int = 0

    # Keep only the newest N checkpoints (0 disables deletion).
    checkpoint_rotate: int = 0

    # Optional: subdirectory name under run_dir to store checkpoints.
    checkpoint_subdir: str = "checkpoints"

    # Optional: include parameter banks in periodic checkpoints (recommended for determinism).
    checkpoint_include_banks: bool = True


    # -----------------------------
    # Phase 6 (draft): Pressure Autopilot
    # -----------------------------
    # The autopilot is a deterministic controller that schedules ControlPlane
    # PressureUpdateCommand interventions at SafeBoundary based on:
    # - cohort survival rate
    # - genome diversity collapse metrics
    # - Phase 5 EPI + AV
    #
    # It is OFF by default. Turn it on via config or at runtime using a
    # PressureUpdateCommand that sets sim_cfg.pressure_autopilot_enable=true.
    pressure_autopilot_enable: bool = False
    # When True, the autopilot will log decisions but never schedule commands.
    pressure_autopilot_hold: bool = False
    # When True, the autopilot logs decisions but never schedules commands (Option C).
    pressure_autopilot_suggestion_only: bool = False
    # Profile selects which knobs are controlled:
    #   balanced | cycle | scarcity | cost | shock | scripted
    pressure_autopilot_profile: str = "balanced"
    # Optional scripted-autopilot spec (dict) for deterministic one-shot or
    # pulse interventions that still flow through the autopilot -> ControlPlane
    # -> SafeBoundary apply path.
    pressure_autopilot_script: dict[str, object] = field(default_factory=dict)
    # Generation interval between controller ticks (SafeBoundary generations).
    pressure_autopilot_tick_interval_gen: int = 500
    # Minimum SafeBoundary generation to start controlling.
    pressure_autopilot_start_gen: int = 0
    # Commands are scheduled for generation_index + apply_delay.
    # Must be >= 1 because SafeBoundary already processed commands for this generation.
    pressure_autopilot_apply_delay_gen: int = 1

    # Pressure scalar in [0, 1]. 0=easiest, 1=hardest.
    pressure_autopilot_pressure_init: float = 0.50

    # P-controller knobs for pressure_level updates.
    pressure_autopilot_target_survival: float = 0.55
    pressure_autopilot_survival_deadband: float = 0.05
    pressure_autopilot_survival_floor: float = 0.20
    pressure_autopilot_survival_ceiling: float = 0.80
    pressure_autopilot_delta_up: float = 0.05
    pressure_autopilot_delta_down: float = 0.10

    # Diversity: if diversity falls below this floor and survival is also low,
    # the controller eases pressure slightly.
    pressure_autopilot_diversity_floor: float = 0.0

    # AV thresholds (AV is None until long_window generations).
    pressure_autopilot_av_low: float = -0.25
    pressure_autopilot_av_high: float = 0.25
    pressure_autopilot_av_deadband: float = 0.05

    # Cycle profile: switch axis when EPI fails to improve for N controller ticks.
    pressure_autopilot_epi_improve_eps: float = 0.25
    pressure_autopilot_stagnation_ticks_required: int = 2
    # Or switch axis when diversity collapse streak gets too long.
    pressure_autopilot_collapse_streak_switch: int = 3

    # Float comparison epsilon for deciding whether to schedule a knob update.
    pressure_autopilot_float_eps: float = 1e-6

    # If you set shock_start_prob to 0.0 intentionally, autopilot will respect
    # that and keep it at 0.0 unless you explicitly allow re-enabling.
    pressure_autopilot_allow_shock_enable: bool = False
    pressure_autopilot_shock_baseline_if_zero: float = 0.02

    # If True, baselines for controlled knobs are reset the first time the
    # autopilot runs (useful when enabling autopilot mid-run).
    pressure_autopilot_reset_on_enable: bool = False


    # -----------------------------
    # Phase 3: Checkpoint Bundle v2 (portable, canonical JSON)
    # -----------------------------
    # Save a portable checkpoint bundle every K generations (0 disables).
    # This is independent of legacy Checkpoint v1.
    checkpoint_bundle_v2_period_gen: int = 0

    # Optional: subdirectory name under run_dir to store v2 bundles.
    checkpoint_bundle_v2_subdir: str = "checkpoint_bundles_v2"

    # Persist strict replay intent into bundle metadata. Loader enforces strict mode gates.
    checkpoint_bundle_v2_strict_replay_mode: bool = True



    # -----------------------------
    # Phase 1.5: knob override audit trail
    # -----------------------------
    # Canonical list of applied overrides (section.field=value). Persisted to config.json.
    knob_overrides: list[str] = field(default_factory=list)

    # Genome overrides template (field -> raw string). Applied to Genome instances at spawn time.
    genome_overrides: dict[str, str] = field(default_factory=dict)

    # Final post-profile readback for accepted generic config overrides.
    knob_override_receipts: list[dict[str, Any]] = field(default_factory=list)


    # -----------------------------
    # Feature: Identity Gate (Phase 4b)
    # -----------------------------
    identity_gate_enable: bool = False
    identity_gate_mode: str = "temperature"
    identity_gate_strength: float = 0.50
    identity_gate_adult_only: bool = True
    identity_gate_temp_floor: float = 0.05

    # -----------------------------
    # Feature: Developmental Plasticity Decay (Phase 4b)
    # -----------------------------
    dev_plasticity_decay_enable: bool = False
    dev_plasticity_half_life_steps: int = 200


    # -----------------------------
    # Logging control
    # -----------------------------
    # If False, BehaviorLogger is replaced with a no-op implementation.
    # Default True preserves legacy behavior logging.
    behavior_log_enable: bool = True

    # -----------------------------
    # Track C2: Prediction-demand action-context reporting
    # -----------------------------
    # Reporting-only and default-off. These flags do not alter action choice,
    # reward, selection, reproduction, memory state, or prediction state.
    prediction_demand_action_context_enable: bool = False
    prediction_demand_action_context_schema_version: str = "prediction_demand_action_context_v1"
    prediction_demand_action_context_window_radius: int = 1
    prediction_demand_action_context_include_lineage: bool = True
    prediction_demand_action_context_write_events: bool = False
    prediction_demand_action_context_max_events_per_run: int = 0
    prediction_demand_action_context_sample_tick_period: int = 1
    prediction_demand_action_context_sample_generation_period: int = 1
    prediction_demand_action_context_sample_first_n_generations: int = 0
    prediction_demand_action_context_run_id: str = ""
    prediction_demand_action_context_memory_arm: str = ""
    prediction_demand_action_context_prediction_arm: str = ""
    prediction_demand_action_context_init_arm: str = ""
    prediction_demand_action_context_sign_summary_enable: bool = False
    prediction_demand_action_context_sign_summary_schema_version: str = (
        "prediction_demand_action_context_sign_summary_v1"
    )
    prediction_demand_action_context_sign_summary_include_windows: bool = True
    prediction_demand_action_context_sign_summary_write_jsonl: bool = True
    prediction_demand_action_context_sign_summary_write_compact_csv: bool = False
    prediction_demand_action_context_sign_summary_max_events_per_run: int = 0


    # -----------------------------
    # Policy sampling control
    # -----------------------------
    # If True, policy action selection uses deterministic argmax (no sampling noise).
    # Default False preserves stochastic sampling behavior.
    policy_deterministic_actions: bool = False
    mosaic_fixed1_probe_enable: bool = False
    mosaic_fixed1_probe_schema_version: int = 1
    mosaic_fixed1_probe_arm: str = ""
    mosaic_fixed1_probe_controller_version: str = "mosaic_fixed1_reactive_v1"
    mosaic_fixed1r_telemetry_enable: bool = False
    mosaic_fixed1r_telemetry_schema_version: int = 1
    predrisk_fixed1_probe_enable: bool = False
    predrisk_fixed1_probe_schema_version: int = 1
    predrisk_fixed1_probe_arm: str = ""
    predrisk_fixed1_probe_controller_version: str = "predrisk_fixed1_mlp_v1"
    predrisk_fixed1_telemetry_enable: bool = False
    predrisk_fixed1_telemetry_schema_version: int = 1


    # -----------------------------
    # Phase 5a: Experimental Capability Metrics (EPI + AV)
    # -----------------------------
    epi_enable: bool = False

    # Scenario selection:
    # - Empty list => engine uses built-in defaults.
    # - Non-empty list => filter defaults to these scenario_ids.
    epi_scenario_ids: list[str] = field(default_factory=list)

    # Population sampling policy:
    # - "all": evaluate entire population
    # - "fixed_stride": deterministic subsample by agent_id order
    epi_subsample_mode: str = "all"
    epi_subsample_stride: int = 1
    epi_subsample_start_index: int = 0

    # Scenario score weights (ws1+ws2+ws3 SHOULD sum to 1.0; engine renormalizes if not).
    epi_ws1_survival: float = 0.50
    epi_ws2_reproduction: float = 0.30
    epi_ws3_robustness: float = 0.20

    # Aggregation weights.
    epi_alpha_mean: float = 0.70
    epi_alpha_min: float = 0.30

    # AV settings (computed only if epi_enable).
    av_enable: bool = True
    av_smoothing_k: int = 5
    av_short_window: int = 20
    av_long_window: int = 100

    # Phase 5a follow-up: optional EPI/AV visibility (human-friendly).
    # - epi_print_each_gen: prints one line per generation to stdout
    # - epi_timeseries_enable: writes epi_av_timeseries.csv (append-only) in run_dir
    epi_print_each_gen: bool = False
    epi_timeseries_enable: bool = False

    # -----------------------------
    # Phase E1: deterministic evolvability metrics
    # -----------------------------
    phase_e_e1_enable: bool = True
    phase_e_e1_cadence_gen: int = 1
    phase_e_e1_retention_k_gen: int = 3
    phase_e_changed_agent_hash_reuse_enable: bool = False

    # -----------------------------
    # Theme 4 Stage 0: evolvable cognition framework
    # -----------------------------
    # The trait registry is static and always available. This deprecated field
    # is kept only for run-config compatibility; it does not gate construction.
    enable_cognitive_trait_registry: bool = True
    # These flags enable deterministic metadata/telemetry only. They do not add
    # agent intelligence, variable architecture, or new energy charging.
    enable_cognitive_status_metrics: bool = True
    enable_cognitive_budget_ledger: bool = True
    trait_capacity_saturation_reporting_enable: bool = False
    enable_experimental_cognitive_cost_charging: bool = False
    native_cognitive_budget_cost_aggregation_enable: bool = False
    native_cognitive_budget_cost_aggregation_schema_version: int = 1
    native_cognitive_budget_cost_aggregation_fallback_mode: str = "numpy"
    native_cognitive_budget_cost_aggregation_diagnostics_enable: bool = True
    native_cognitive_budget_cost_aggregation_mismatch_guard: bool = False
    env_hotpath_direct_cost_scalar_enable: bool = False
    env_hotpath_direct_cost_scalar_schema_version: int = 1
    env_hotpath_direct_cost_scalar_mismatch_guard: bool = False
    env_hotpath_direct_cost_scalar_guard_mode: str = "sample"
    env_hotpath_direct_cost_scalar_guard_sample_period: int = 1000
    env_hotpath_cost_charge_surface_enable: bool = False
    env_hotpath_cost_charge_surface_schema_version: int = 1
    env_hotpath_cost_charge_surface_guard_enable: bool = False
    env_hotpath_cost_charge_surface_guard_mode: str = "sample"
    env_hotpath_cost_charge_surface_guard_sample_period: int = 1000
    env_hotpath_costsurface_soa_diagnostics_enable: bool = False
    cxxcore6_authoritative_pilot_enable: bool = False
    cxxcore6_authoritative_no_rng_only: bool = True
    cxxcore6_authoritative_compare_mode: str = "off"
    cxxcore6_authoritative_fallback_on_divergence: bool = True
    cxxcore6_authoritative_fail_fast_on_divergence: bool = False
    cxxcore6_force_python_baseline: bool = False
    cxxcore6_native_required: bool = False
    cxxcore7_rng_precompute_enable: bool = False
    cxxcore7_validation_action_override_enable: bool = False
    cxxcore7_runner_bridge_probe_enable: bool = False
    cxxcore7_native_required: bool = False
    cxxcore8_guarded_runner_bridge_enable: bool = False
    cxxcore8_guarded_runner_bridge_mode: str = "shadow"
    cxxcore8_runner_bridge_compare_interval_ticks: int = 0
    cxxcore8_runner_bridge_force_final_compare: bool = True
    cxxcore8_runner_bridge_fallback_on_mismatch: bool = True
    cxxcore8_runner_bridge_emit_compact_summary: bool = True
    cxxcore8_runner_bridge_allow_validation_rng_precompute: bool = False
    cxxcore8_runner_bridge_allow_biome_hazard_precompute: bool = False
    cxxcore8_runner_bridge_allow_tier3_precompute: bool = False
    cxxcore9_runner_adapter_enable: bool = False
    cxxcore9_runner_adapter_mode: str = "off"
    cxxcore9_runner_adapter_parity_interval: int = 1
    cxxcore9_runner_adapter_restore_on_divergence: bool = True
    cxxcore9_runner_adapter_emit_diagnostics: bool = True
    cxxcore9_runner_adapter_fail_closed: bool = True
    cxxcore9_runner_adapter_strict_native_required: bool = False
    cxxcore9_runner_adapter_parity_policy: str = "every_tick"
    cxxcore10_env_tick_split_enable: bool = False
    cxxcore10_env_tick_split_mode: str = "off"
    cxxcore10_env_tick_split_parity_interval: int = 1
    cxxcore10_env_tick_split_restore_on_divergence: bool = True
    cxxcore10_env_tick_split_emit_diagnostics: bool = True
    cxxcore10_env_tick_split_fail_closed: bool = True
    cxxcore10_env_tick_split_strict_native_required: bool = False
    cxxcore10_env_tick_split_parity_policy: str = "final_and_sparse"
    cxxcore11_live_runner_hook_enable: bool = False
    cxxcore11_live_runner_hook_mode: str = "off"
    cxxcore11_live_runner_hook_parity_interval: int = 1
    cxxcore11_live_runner_hook_restore_on_divergence: bool = True
    cxxcore11_live_runner_hook_fail_closed: bool = True
    cxxcore11_live_runner_hook_emit_diagnostics: bool = True
    cxxcore11_live_runner_hook_strict_native_required: bool = False
    cxxcore11_live_runner_hook_parity_policy: str = "final_and_sparse"
    cxxcore11_live_runner_hook_rng_policy: str = "none"
    cxxcore12_live_tick_island_enable: bool = False
    cxxcore12_live_tick_island_mode: str = "off"
    cxxcore12_live_tick_island_parity_policy: str = "final_and_sparse"
    cxxcore12_live_tick_island_parity_interval: int = 1
    cxxcore12_live_tick_island_restore_on_divergence: bool = True
    cxxcore12_live_tick_island_fail_closed: bool = True
    cxxcore12_live_tick_island_strict_native_required: bool = False
    cxxcore12_live_tick_island_rng_policy: str = "none"
    cxxcore12_live_tick_island_emit_diagnostics: bool = True
    cxxcore13_state_backend_enable: bool = False
    cxxcore13_state_backend_mode: str = "off"
    cxxcore13_state_backend_parity_policy: str = "final_and_sparse"
    cxxcore13_state_backend_parity_interval: int = 1
    cxxcore13_state_backend_restore_on_divergence: bool = True
    cxxcore13_state_backend_fail_closed: bool = True
    cxxcore13_state_backend_strict_native_required: bool = False
    cxxcore13_state_backend_rng_policy: str = "none"
    cxxcore13_state_backend_emit_diagnostics: bool = True
    cxxcore16_resident_window_enable: bool = False
    cxxcore16_resident_window_authoritative_enable: bool = False
    cxxcore16_resident_window_mode: str = "off"
    cxxcore16_resident_window_compare_enable: bool = True
    cxxcore16_resident_window_sync_interval: int = 0
    cxxcore16_resident_window_max_ticks: int = 0
    cxxcore16_resident_window_allow_validation_rng_precompute: bool = False
    cxxcore16_resident_window_fail_closed: bool = True
    cxxcore16_resident_window_restore_on_divergence: bool = True
    cxxcore17_runner_island_enable: bool = False
    cxxcore17_runner_island_mode: str = "off"
    cxxcore17_runner_island_compare_enable: bool = True
    cxxcore17_runner_island_summary_interval: int = 0
    cxxcore17_runner_island_fail_closed: bool = True
    cxxcore_summary_tape_enable: bool = False
    cxxcore_summary_tape_shadow_compare_enable: bool = False
    cxxcore_summary_tape_authoritative_metrics_enable: bool = False
    cxxcore_summary_tape_consumers: tuple[str, ...] = ()
    cxxcore_summary_tape_first_divergence_limit: int = 1
    cxxcore_summary_tape_mode: str = "off"
    cxxcore_direct_compact_diagnostics_enable: bool = False
    cxxcore_direct_compact_diagnostics_mode: str = "off"
    cxxcore_direct_compact_diagnostics_guarded: bool = False
    cxxcore_direct_compact_diagnostics_compare_every_n: int = 0
    cxxcore_direct_compact_diagnostics_force_native_missing: bool = False
    cxxcore_direct_compact_diagnostics_force_native_exception: bool = False
    cxxcore_direct_compact_diagnostics_inject_divergence: bool = False

    # -----------------------------
    # Theme 4 Stage 1A/1B: evolvable perception budget
    # -----------------------------
    # Defaults preserve existing behavior: genes are serialized and measured, but
    # observation masking, new mutations, and new energy charging are off unless
    # these explicit experimental flags are enabled.
    enable_evolvable_sensor_channel_mask: bool = False
    sensor_channel_mask_mutation_rate: float = 0.02
    sensor_channel_preserve_basic_channel: bool = True
    ablate_sensor_channel_mask: bool = False
    force_all_sensor_channels_enabled: bool = False
    force_minimal_sensor_channels_enabled: bool = False
    sensor_channel_cost_per_weight: float = 0.001
    enable_experimental_perception_cost_charging: bool = False

    enable_evolvable_sensor_radius: bool = False
    sensor_radius_mutation_rate: float = 0.05
    ablate_sensor_radius: bool = False
    force_min_sensor_radius: bool = False
    force_max_sensor_radius: bool = False
    force_default_sensor_radius: bool = False
    sensor_radius_cost_base: float = 0.0
    sensor_radius_cost_per_cell: float = 0.0001
    sensor_radius_cost_per_cell_channel: float = 0.00005
    enable_experimental_sensor_radius_cost_charging: bool = False

    # -----------------------------
    # Theme 4 Stage 2: memory observability and ablation
    # -----------------------------
    # These flags are runtime expression overrides. They do not mutate genomes and
    # they do not introduce variable GRU hidden shapes.
    ablate_memory: bool = False
    force_memory_disabled: bool = False
    force_min_memory_dim: bool = False
    force_max_memory_dim: bool = False
    force_default_memory_dim: bool = False
    reset_memory_each_tick: bool = False
    enable_experimental_memory_cost_charging: bool = False
    memory_pressure_scenario: bool = False
    memory_pressure_type: str = ""

    # -----------------------------
    # Theme 4 Stage 3: prediction observability and ablation
    # -----------------------------
    ablate_prediction: bool = False
    force_prediction_disabled: bool = False
    force_prediction_frozen: bool = False
    force_prediction_zero_lr: bool = False
    force_prediction_default: bool = False
    reset_prediction_each_tick: bool = False
    disable_prediction_error_selection_pressure: bool = False
    enable_experimental_prediction_cost_charging: bool = False
    prediction_cost_base: float = 0.0
    prediction_cost_per_lr: float = 0.001
    prediction_state_cost: float = 0.0
    prediction_pressure_scenario: bool = False
    prediction_pressure_type: str = ""

    # -----------------------------
    # Theme 4 Stage 4: developmental cognition observability and ablation
    # -----------------------------
    enable_developmental_cognition: bool = False
    developmental_cognition_mode: str = "legacy_full_expression"
    developmental_maturity_age: int = 20
    life_history_profile_id: str = "legacy_inert_v1"
    life_history_mode: int = 0
    life_history_maturity_start_ticks: int = 0
    life_history_maturity_rise_ticks: int = 0
    life_history_fertility_decline_start_ticks: int = 0
    life_history_fertility_decline_ticks: int = 0
    life_history_elder_fertility_floor: float = 1.0
    life_history_senescence_start_ticks: int = 0
    life_history_senescence_ramp_ticks: int = 0
    life_history_senescence_energy_cost: float = 0.0
    life_history_senescence_health_cost: float = 0.0
    life_history_max_age_ticks: int = 0
    juvenile_cognition_scale: float = 0.5
    adult_cognition_scale: float = 1.0
    ablate_developmental_cognition: bool = False
    force_adult_cognition: bool = False
    force_juvenile_cognition: bool = False
    force_legacy_full_cognition: bool = False
    disable_cognitive_growth: bool = False
    force_zero_maturation_cost: bool = False
    force_max_maturation_cost: bool = False
    enable_experimental_developmental_cost_charging: bool = False
    developmental_cognition_cost_base: float = 0.0
    developmental_cognition_cost_per_unexpressed_capacity: float = 0.001
    maturation_cost_per_tick: float = 0.0
    developmental_pressure_scenario: bool = False
    developmental_pressure_type: str = ""

    # -----------------------------
    # Theme 4 Stage 5: introspection-ready internal-state channels
    # -----------------------------
    enable_introspection_channels: bool = False
    default_introspection_channels_enabled: bool = False
    enable_evolvable_introspection_channels: bool = False
    enable_introspection_channel_mutation: bool = False
    enable_introspection_channel_recombination: bool = False
    introspection_channel_mask_mutation_rate: float = 0.02
    enable_introspection_policy_input: bool = False
    enable_experimental_introspection_cost_charging: bool = False
    introspection_channel_cost: float = 0.001
    introspection_observation_cost_base: float = 0.0
    ablate_introspection: bool = False
    force_no_introspection_channels: bool = False
    force_minimal_introspection_channels: bool = False
    force_all_introspection_channels: bool = False
    force_default_introspection_channels: bool = False
    mask_introspection_values_to_neutral: bool = False
    introspection_pressure_scenario: bool = False
    introspection_pressure_type: str = ""

    # -----------------------------
    # Theme 4 Stage 6: bounded self-regulation surface
    # -----------------------------
    enable_self_regulation: bool = False
    enable_self_regulation_surface: bool = False
    default_self_regulation_controls_enabled: bool = False
    enable_evolvable_self_regulation_controls: bool = False
    enable_self_regulation_control_mutation: bool = False
    enable_self_regulation_control_recombination: bool = False
    self_regulation_control_mask_mutation_rate: float = 0.02
    enable_self_regulation_policy_output: bool = False
    enable_experimental_self_regulation_cost_charging: bool = False
    self_regulation_control_cost_base: float = 0.0
    self_regulation_control_cost: float = 0.001
    self_regulation_update_cost_scale: float = 0.001
    self_regulation_offset_decay: float = 0.05
    ablate_self_regulation: bool = False
    force_no_self_regulation: bool = False
    force_all_self_regulation_controls: bool = False
    force_default_self_regulation_controls: bool = False
    mask_self_regulation_outputs_to_neutral: bool = False
    freeze_self_regulation_state: bool = False
    reset_self_regulation_each_tick: bool = False
    disable_self_regulation_decay: bool = False
    self_regulation_pressure_scenario: bool = False
    self_regulation_pressure_type: str = ""

    # -----------------------------
    # Theme 4 Stage 7: bounded self-modification surface
    # -----------------------------
    enable_self_modification: bool = False
    default_self_modification_targets_enabled: bool = False
    enable_evolvable_self_modification_targets: bool = False
    enable_self_modification_target_mutation: bool = False
    enable_self_modification_target_recombination: bool = False
    self_modification_target_mask_mutation_rate: float = 0.02
    enable_self_modification_policy_output: bool = False
    enable_experimental_self_modification_cost_charging: bool = False
    self_modification_command_cost_base: float = 0.0
    self_modification_target_cost: float = 0.001
    self_modification_update_cost_scale: float = 0.001
    self_modification_offset_decay: float = 0.01
    ablate_self_modification: bool = False
    force_no_self_modification: bool = False
    force_all_self_modification_targets: bool = False
    force_default_self_modification_targets: bool = False
    mask_self_modification_commands_to_neutral: bool = False
    freeze_self_modification_state: bool = False
    reset_self_modification_each_tick: bool = False
    reset_self_modification_on_reproduction: bool = False
    disable_self_modification_decay: bool = False
    self_modification_pressure_scenario: bool = False
    self_modification_pressure_type: str = ""

    # -----------------------------
    # Theme 4 Stage 8: directed-inheritance groundwork
    # -----------------------------
    enable_directed_inheritance: bool = False
    enable_directed_inheritance_surface: bool = False
    default_directed_inheritance_channels_enabled: bool = False
    enable_evolvable_directed_inheritance_channels: bool = False
    enable_directed_inheritance_channel_mutation: bool = False
    enable_directed_inheritance_channel_recombination: bool = False
    directed_inheritance_channel_mask_mutation_rate: float = 0.02
    enable_directed_inheritance_policy_output: bool = False
    enable_experimental_directed_inheritance_cost_charging: bool = False
    directed_inheritance_command_cost_base: float = 0.0
    directed_inheritance_channel_cost: float = 0.001
    directed_inheritance_update_cost_scale: float = 0.001
    ablate_directed_inheritance: bool = False
    force_no_directed_inheritance: bool = False
    force_all_directed_inheritance_channels: bool = False
    force_default_directed_inheritance_channels: bool = False
    mask_directed_inheritance_commands_to_neutral: bool = False
    freeze_directed_inheritance_payload: bool = False
    reset_directed_inheritance_payload_on_birth: bool = False
    reset_directed_inheritance_payload_on_maturity: bool = False
    disable_directed_inheritance_costs: bool = False
    directed_inheritance_pressure_scenario: bool = False
    directed_inheritance_pressure_type: str = ""
    phase_e_e1_neutrality_window: int = 8
    phase_e_e1_knockout_every_gen: int = 2
    phase_e_e1_knockout_max_agents: int = 16
    phase_e_e1_knockout_max_modules_per_agent: int = 4

    # -----------------------------
    # Phase E2: deterministic failure taxonomy
    # -----------------------------
    phase_e_e2_enable: bool = False
    phase_e_e2_cadence_gen: int = 1
    phase_e_e2_drift_collapse_min_streak: int = 2
    phase_e_e2_drift_collapse_min_ratio: float = 0.50
    phase_e_e2_structural_stagnation_min_streak: int = 5
    phase_e_e2_bloat_cap_ratio: float = 0.95
    phase_e_e2_mutation_explosion_ratio: float = 0.75
    phase_e_e2_identity_mimicry_max_ratio: float = 0.15

# -----------------------------
# Legacy / general coherence penalty knobs
# -----------------------------
coherence_thrashing_weight: float = 0.15
coherence_drive_jitter_weight: float = 0.08
coherence_selfharm_weight: float = 3.0
coherence_pred_err_weight: float = 1.25
