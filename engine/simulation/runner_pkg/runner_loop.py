# simulation/runner_pkg/loop.py
# Last modified: 2026-02-23 (Europe/Vienna)
from __future__ import annotations

import inspect
import time
import os
import signal
from pathlib import Path
from typing import Dict

import numpy as np

from simulation.metrics import ProgressLogger
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_save, rotate_checkpoint_bundles_v2
from simulation.safeboundary_hash_log import compute_safeboundary_hashes, append_safeboundary_hash_log
from simulation.founder_bank_artifact import (
    maybe_apply_founder_bank_artifact,
    maybe_place_founder_bank_artifact_startup,
)
from simulation.metrics_pkg.lineage_reproduction_funnel import (
    capture_pre_turnover_snapshot as _capture_lineage_funnel_pre_turnover,
    enabled as _lineage_funnel_enabled,
    write_generation as _write_lineage_funnel_generation,
)
from simulation.metrics_pkg.structural_selection_funnel import (
    capture_pre_turnover_snapshot as _capture_structural_selection_funnel_pre_turnover,
    enabled as _structural_selection_funnel_enabled,
    write_generation as _write_structural_selection_funnel_generation,
)
from simulation.metrics_pkg.structural_tick_reproduction_funnel import (
    enabled as _structural_tick_reproduction_funnel_enabled,
    write_generation as _write_structural_tick_reproduction_funnel_generation,
)


def _perf_cuda_sync_if_enabled(sim) -> None:
    """Synchronize CUDA for opt-in diagnostic timings only."""
    try:
        if not bool(getattr(sim, "perf", False)):
            return
        sim_cfg = getattr(sim, "sim_cfg", None)
        if not bool(getattr(sim_cfg, "perf_sync_cuda", False)):
            return
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _perf_time(sim) -> float:
    _perf_cuda_sync_if_enabled(sim)
    return time.perf_counter()


def _perf_add(sim, buckets: Dict[str, float], bucket: str, started_at: float) -> None:
    elapsed = _perf_elapsed(sim, started_at)
    buckets[bucket] = float(buckets.get(bucket, 0.0)) + elapsed


def _perf_elapsed(sim, started_at: float) -> float:
    _perf_cuda_sync_if_enabled(sim)
    return time.perf_counter() - started_at


def _perf_add_checkpoint_details(sim, buckets: Dict[str, float]) -> None:
    try:
        detail = getattr(sim, "_last_checkpoint_perf", None)
        if not isinstance(detail, dict):
            return
        for key in (
            "checkpoint_serialize",
            "checkpoint_hash",
            "checkpoint_write",
            "checkpoint_atomic_replace",
        ):
            buckets[key] = float(buckets.get(key, 0.0)) + float(detail.get(key, 0.0) or 0.0)
        setattr(sim, "_last_checkpoint_perf", None)
    except Exception:
        return


def _write_heartbeat(run_dir: str, generation_index: int) -> None:
    """Write an operational heartbeat file (non-deterministic metadata only).

    The heartbeat is used by the WebUI control plane to detect liveness
    and does not participate in simulation state or checkpoints.
    """
    try:
        import json
        import time
        import os
        p = os.path.join(str(run_dir), "heartbeat.json")
        obj = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gen": int(generation_index)}
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, sort_keys=True, separators=(",", ":"))
            f.write("\n")
        os.replace(tmp, p)
    except Exception:
        return


def _route_neutral_typed_lifecycle_output_enabled(sim) -> bool:
    return bool(
        getattr(
            getattr(sim, "sim_cfg", None),
            "typed_lifecycle_public_output_route_neutral_enable",
            False,
        )
    )


class _NativeActiveCheckpointFailure(RuntimeError):
    pass


class _CapacityEpochMigrationFailure(RuntimeError):
    pass


class _CapacityEpochMonitoringFailure(RuntimeError):
    pass


def _native_runtime_active(sim) -> bool:
    decision = getattr(sim, "_native_safe_default_decision", None)
    if isinstance(decision, dict) and "native_execution_active" in decision:
        return bool(decision["native_execution_active"])
    from native_runtime.safe_default import native_generation_requested  # noqa: WPS433

    return bool(native_generation_requested(sim))


def _raise_native_checkpoint_failure(sim, exc: BaseException, *, stage: str) -> None:
    if not _native_runtime_active(sim):
        return
    from native_runtime.runner_integration import record_native_runtime_failure  # noqa: WPS433

    record_native_runtime_failure(sim, exc, stage=stage)
    raise _NativeActiveCheckpointFailure(
        f"native-active checkpoint write failed at {stage}: {exc}"
    ) from exc


def _prepare_route_neutral_typed_lifecycle_output(sim) -> None:
    if not _route_neutral_typed_lifecycle_output_enabled(sim):
        return
    from perf.cxxbackbone10_resident_state_owner import (  # noqa: WPS433
        install_reference_committed_death_oracle_hook,
    )

    install_reference_committed_death_oracle_hook()


def _emit_route_neutral_typed_lifecycle_output(sim) -> None:
    if not _route_neutral_typed_lifecycle_output_enabled(sim):
        return
    from native_runtime.runner_integration import (  # noqa: WPS433
        emit_route_neutral_typed_lifecycle_public_outputs,
    )

    try:
        output = emit_route_neutral_typed_lifecycle_public_outputs(sim)
        if not bool(output.get("flush_complete", False)):
            raise RuntimeError("route-neutral typed lifecycle output did not flush successfully")
        summary = getattr(sim, "_native_runtime_summary", None)
        if _native_runtime_active(sim) and isinstance(summary, dict):
            validation = dict(summary.get("native_result_validation") or {})
            validation["typed_lifecycle_output_flush_complete"] = True
            validation["output_flush_complete"] = bool(
                validation.get("behavior_output_flush_complete", False)
            )
            validation["public_output_flush_pending"] = not validation["output_flush_complete"]
            summary["native_result_validation"] = validation
            setattr(sim, "_native_runtime_summary", summary)
    except Exception as exc:
        if _native_runtime_active(sim):
            from native_runtime.runner_integration import record_native_runtime_failure  # noqa: WPS433

            record_native_runtime_failure(sim, exc, stage="typed_lifecycle_public_output")
        raise


def _try_execute_cuda_architecture_b_run(
    sim,
    *,
    start_gen: int,
    end_gen: int,
    skip_init: bool,
):
    """Hand one qualified run to the native CUDA owner when its envelope matches."""

    if not bool(
        getattr(sim.sim_cfg, "cuda_architecture_b_production_enable", False)
    ):
        return None
    mode = str(getattr(sim.sim_cfg, "cuda_architecture_b_request_mode", "auto"))
    try:
        from native_runtime.cuda_architecture_b_backend_factory import (  # noqa: WPS433
            derive_cuda_architecture_b_production_runner_envelope,
        )
        from native_runtime.cuda_architecture_b_routing import (  # noqa: WPS433
            explicit_legacy_python_pre_dispatch_receipt,
            native_engine_pre_dispatch_rejection_receipt,
        )

        derive_cuda_architecture_b_production_runner_envelope(sim)
    except Exception as exc:
        reason = (
            "outside_qualified_cuda_envelope:"
            f"{type(exc).__name__}:{exc}"
        )
        if mode == "auto":
            sim._cuda_architecture_b_route_receipt = (
                explicit_legacy_python_pre_dispatch_receipt(reason=reason)
            )
            return None
        sim._cuda_architecture_b_route_receipt = (
            native_engine_pre_dispatch_rejection_receipt(
                reason=reason,
                request_mode=mode,
            )
        )
        raise RuntimeError(
            "EIEL-NATIVE-ENGINE-PRE-DISPATCH-REJECTED: " + reason
        ) from exc
    from simulation.tick_native_sampling import science_sampling_enabled  # noqa: WPS433

    if science_sampling_enabled(sim.sim_cfg):
        raise RuntimeError(
            "TN-2 scientific sampling is not qualified for the CUDA native owner; "
            "use the reference CPU runner or disable sampling"
        )
    from native_runtime.cuda_architecture_b_runner_owner_v1 import (  # noqa: WPS433
        execute_cuda_architecture_b_runner_run_v1,
    )

    return execute_cuda_architecture_b_runner_run_v1(
        sim,
        start_gen=int(start_gen),
        end_gen=int(end_gen),
        skip_init=bool(skip_init),
    )


class LoopMixin:
    def run(self, *, start_gen: int = 0, skip_init: bool = False):
        """Run the simulation loop.

        Parameters
        ----------
        start_gen:
            Absolute generation index to start from.
            - Default 0 preserves legacy behavior.
            - When resuming from a checkpoint, this should be (checkpoint_gen + 1).
        skip_init:
            If True, skips initial population placement / bank init that would overwrite
            checkpoint-restored state. Intended for resume-from-checkpoint.
        """
        self.pop.sim_cfg = getattr(self, 'sim_cfg', None)
        _prepare_route_neutral_typed_lifecycle_output(self)



        # ------------------------------------------------------------------
        # Determinism probes (Phase 3): alive membership snapshots
        # ------------------------------------------------------------------
        # Purpose:
        #   Localize resume divergence by detecting whether alive membership
        #   differs BEFORE a generation starts.
        #
        # Constraints:
        #   - Best-effort only; must never crash production runs.
        #   - No RNG draws.
        #   - No bundle schema changes.
        # ------------------------------------------------------------------

        def _maybe_dump_alive_probe(*, generation_index: int, tag: str) -> None:
            """Best-effort deterministic dump of pop/env alive membership.

            Notes
            -----
            Uses sorted ID lists for stable ordering. The dump is gated by
            EIEL_STATE_DUMP and EIEL_STATE_DUMP_GENS via maybe_dump_state.
            """
            # Import must work in both repo layouts:
            # - when engine/ is on sys.path, modules are imported as `simulation.*`
            # - in some harnesses the full package path exists
            try:
                from simulation.state_dump import maybe_dump_state as _maybe_dump_state  # type: ignore
            except Exception:
                try:
                    from engine.simulation.state_dump import maybe_dump_state as _maybe_dump_state  # type: ignore  # noqa: WPS433
                except Exception:
                    return
            try:
                # Some env implementations do not maintain an explicit alive mask.
                # We still dump population membership deterministically so the
                # membership harness can operate on pop-only evidence.
                env_alive = getattr(self.env, "agent_alive", None)
                cap = None
                if env_alive is not None:
                    try:
                        cap = int(env_alive.shape[0])
                    except Exception:
                        cap = None

                pop_alive_ids = []
                for a in self.pop.alive_agents():
                    try:
                        aid = int(a.agent_id)
                    except Exception:
                        continue
                    if (cap is None) or (0 <= aid < cap):
                        pop_alive_ids.append(aid)

                pop_alive_ids_sorted = sorted(pop_alive_ids)
                env_alive_ids_sorted = []
                if env_alive is not None:
                    env_alive_ids_sorted = [int(i) for i in np.nonzero(env_alive)[0].tolist()]

                pop_set = set(pop_alive_ids_sorted)
                env_set = set(env_alive_ids_sorted)

                extra = {
                    "pop_alive_n": int(len(pop_alive_ids_sorted)),
                    "env_alive_n": int(len(env_alive_ids_sorted)),
                    "pop_not_env_n": int(len(pop_set - env_set)),
                    "env_not_pop_n": int(len(env_set - pop_set)),
                    "alive_ids": pop_alive_ids_sorted,
                    "pop_alive_ids_sorted": pop_alive_ids_sorted,
                    "env_alive_ids_sorted": env_alive_ids_sorted,
                    "pop_not_env_sorted": sorted(list(pop_set - env_set)),
                    "env_not_pop_sorted": sorted(list(env_set - pop_set)),
                }

                # Optional: attach last membership write records (debug-only).
                try:
                    from simulation.membership_trace import drain_writes  # noqa: WPS433
                    extra["membership_writes"] = drain_writes()
                except Exception:
                    pass

                _maybe_dump_state(self, generation_index=int(generation_index), tag=str(tag), extra=extra)
            except Exception:
                return


        # -----------------------------
        # Graceful interrupt handling (Ctrl-C / SIGINT)
        # -----------------------------
        # IMPORTANT: Do NOT write checkpoints inside the signal handler.
        # The handler only requests a stop; we checkpoint and exit at a safe boundary.
        self._stop_requested = False
        self._sigint_count = 0

        # If we are resuming, allow checkpoints to reflect the "position before this run starts".
        try:
            setattr(self, "_current_gen", int(start_gen) - 1)
        except Exception:
            setattr(self, "_current_gen", -1)

        old_sigint = signal.getsignal(signal.SIGINT)
        old_sigterm = signal.getsignal(signal.SIGTERM)

        def _request_stop(signum, frame):
            """Request a graceful stop at the next safe boundary.

            First interrupt: set a flag and print a one-time message.
            Second interrupt: restore default behavior and re-raise to exit immediately.
            """
            try:
                self._sigint_count += 1
                if self._sigint_count >= 2:
                    # User really means it. Restore defaults and re-raise immediately.
                    try:
                        signal.signal(signal.SIGINT, signal.SIG_DFL)
                    except Exception:
                        pass
                    try:
                        signal.raise_signal(signal.SIGINT)
                    except Exception:
                        raise KeyboardInterrupt
                    return
                self._stop_requested = True
                print("[interrupt] stop requested (Ctrl-C). Will checkpoint + exit at next safe boundary...", flush=True)
            except Exception:
                # Best-effort only. Never crash due to interrupt bookkeeping.
                pass

        try:
            signal.signal(signal.SIGINT, _request_stop)
            signal.signal(signal.SIGTERM, _request_stop)
        except Exception as _sig_exc:
            # Some environments may not allow signal handling (e.g., non-main threads).
            # Do not fail silently: if we cannot install handlers, Ctrl-C may abort mid-generation.
            try:
                print(f"[interrupt] WARNING: could not install SIGINT/SIGTERM handlers: {_sig_exc}", flush=True)
            except Exception:
                pass

        if not skip_init:
            self.pop.init_random()
            if bool(getattr(self.sim_cfg, "recomb72_audit_enable", False)):
                from evolution.recomb72_live_fixture import apply_recomb72_live_founders_v1

                apply_recomb72_live_founders_v1(self)
            founder_bank_artifact = maybe_apply_founder_bank_artifact(self)
            self._ensure_bank()
            if founder_bank_artifact is not None:
                maybe_place_founder_bank_artifact_startup(self, founder_bank_artifact)
            else:
                self._place_agents()
            try:
                bo = getattr(self, "branch_observability", None)
                if bo is not None:
                    bo.register_founders(self.pop.alive_agents(), generation=int(start_gen), tick=0)
            except Exception:
                pass
            if hasattr(self, "_mark_bank_dirty"):
                self._mark_bank_dirty()


        # --------------------------------------------------------------
        # Phase 3 diagnostic support: optional Bundle v2 at simulation start
        # --------------------------------------------------------------
        # Motivation:
        #   Allow a minimal "gen0 -> resume -> gen1" determinism experiment.
        #
        # Contract:
        #   - Only runs for fresh starts (skip_init==False).
        #   - Only when start_gen==0.
        #   - Gated by env var (or an explicit sim_cfg flag) to avoid impacting
        #     default runs.
        #   - Must not draw RNG.
        #
        # Env:
        #   EIEL_CHECKPOINT_BUNDLE_V2_AT_START=1
        # --------------------------------------------------------------
        __eiel_gen0_bundle_enabled = False
        try:
            __eiel_gen0_bundle_enabled = (
                (not bool(skip_init))
                and int(start_gen) == 0
                and (
                    int(os.environ.get("EIEL_CHECKPOINT_BUNDLE_V2_AT_START", "0") or "0") == 1
                    or bool(getattr(self.sim_cfg, "checkpoint_bundle_v2_at_start", False))
                )
            )

            if __eiel_gen0_bundle_enabled:
                # Deterministic breadcrumb so harness logs prove the branch executed.
                try:
                    print("[bundle_v2] gen0 emission enabled", flush=True)
                except Exception:
                    pass

                # Ensure env mirrors population before capturing.
                try:
                    self._sync_env_with_population()
                except Exception:
                    pass

                # Ensure bank rows exist for current agents (no RNG).
                try:
                    alive_now = list(self.pop.alive_agents())
                    self._sync_bank_for_agents(alive_now, buckets=None)
                except Exception:
                    pass

                # Defensive slot-range check.
                bank_cap = int(getattr(self._mlp_pack, "shape", [0])[0]) if getattr(self, "_mlp_pack", None) is not None else 0
                if bank_cap > 0:
                    max_slot = -1
                    for a in self.pop.alive_agents():
                        s = int(getattr(a, "slot", -1))
                        if s > max_slot:
                            max_slot = s
                    if max_slot >= bank_cap:
                        raise RuntimeError(f"gen0 bundle save failed: max_slot={max_slot} bank_cap={bank_cap}")

                v2_subdir = str(getattr(self.sim_cfg, "checkpoint_bundle_v2_subdir", "checkpoint_bundles_v2") or "checkpoint_bundles_v2")
                v2_strict = bool(getattr(self.sim_cfg, "checkpoint_bundle_v2_strict_replay_mode", True))
                ckpt_root = os.path.join(self.run_dir, v2_subdir)
                os.makedirs(ckpt_root, exist_ok=True)
                bpath = os.path.join(ckpt_root, f"bundle_gen{0:06d}")

                checkpoint_bundle_v2_save(self, bpath, generation_index=0, strict_replay_mode=v2_strict)
        except Exception as _gen0_checkpoint_exc:
            # When explicitly enabled, fail hard so harness can surface the real root cause.
            if __eiel_gen0_bundle_enabled:
                _raise_native_checkpoint_failure(
                    self,
                    _gen0_checkpoint_exc,
                    stage="checkpoint_generation_zero",
                )
                raise
            # Otherwise: best-effort only; must never break normal runs.
            pass
        # If an interrupt was requested during initialization (population spawn / placement),
        # we can exit cleanly here before entering the generation loop.
        if getattr(self, "_stop_requested", False):
            # v2-only: emit a Checkpoint Bundle v2 and exit before entering the generation loop.
            try:
                rotate = int(getattr(self.sim_cfg, "checkpoint_rotate", 0) or 0)

                # Prefer v2 subdir; fall back to legacy checkpoint_subdir if provided.
                v2_subdir = str(getattr(self.sim_cfg, "checkpoint_bundle_v2_subdir", "") or "").strip()
                if not v2_subdir:
                    v2_subdir = str(getattr(self.sim_cfg, "checkpoint_subdir", "checkpoint_bundles_v2") or "checkpoint_bundles_v2")
                v2_strict = bool(getattr(self.sim_cfg, "checkpoint_bundle_v2_strict_replay_mode", True))

                ckpt_root = os.path.join(self.run_dir, v2_subdir)
                os.makedirs(ckpt_root, exist_ok=True)

                # _current_gen was set to (start_gen - 1) above.
                gen_before = int(getattr(self, "_current_gen", -1))
                gen_index = int(gen_before) + 1
                bpath = os.path.join(ckpt_root, f"bundle_gen{gen_index:06d}")

                # Ensure env/bank mirrors population before capturing (best-effort, RNG-free).
                try:
                    self._sync_env_with_population()
                except Exception:
                    pass
                try:
                    alive_now = list(self.pop.alive_agents())
                    self._sync_bank_for_agents(alive_now, buckets=None)
                except Exception:
                    pass

                checkpoint_bundle_v2_save(self, bpath, generation_index=int(gen_index), strict_replay_mode=v2_strict)
                rotate_checkpoint_bundles_v2(ckpt_root, keep_last=rotate)
                print(f"[interrupt] checkpoint bundle v2 saved (init): {bpath}", flush=True)
            except Exception as _int_init_exc:
                _raise_native_checkpoint_failure(
                    self,
                    _int_init_exc,
                    stage="checkpoint_init_interrupt",
                )
                try:
                    print(f"[interrupt] WARNING: failed to save init checkpoint bundle v2: {_int_init_exc}", flush=True)
                except Exception:
                    pass
            # Exit without running any generations.
            return


        # NOTE: end_gen must be defined before we initialize any loggers that reference it.
        # When resuming from a checkpoint, start_gen may be > 0. We keep total_gens=end_gen
        # so ProgressLogger can use absolute generation indices (gen) without additional offset.
        end_gen = int(start_gen) + int(self.sim_cfg.generations)

        # -----------------------------
        # Phase 5d: viz/ observability init (UI read-only helpers)
        # -----------------------------
        # These artifacts are intentionally outside the determinism pool.
        _viz = getattr(self, "viz", None)
        _viz_run_kind = "continuation" if bool(getattr(self, "_is_resume_run", False)) else "base"
        _viz_last_completed_gen = int(start_gen) - 1
        if _viz is not None and getattr(_viz, "enabled", False):
            try:
                _viz.initialize_run(
                    start_generation=int(start_gen),
                    target_generation=int(end_gen),
                    run_kind=str(_viz_run_kind),
                )
            except Exception:
                pass

        progress = ProgressLogger(total_gens=end_gen, print_every=1)
        try:
            from monitoring.slot_telemetry import start_runner_slot_lease  # noqa: WPS433

            _slot_lease = start_runner_slot_lease(
                self,
                start_generation=int(start_gen),
                target_generation=int(end_gen),
            )
            _slot_lease.update(
                current_generation=int(start_gen),
                status="running",
                stage="generation loop",
            )
        except Exception:
            _slot_lease = None
        _slot_terminal_status = "failed"
        _slot_terminal_error = ""
        try:
            log_gen_params = set(inspect.signature(self.logger.log_generation).parameters.keys())
        except Exception:
            log_gen_params = set()

        try:
            _cuda_architecture_b_result = _try_execute_cuda_architecture_b_run(
                self,
                start_gen=int(start_gen),
                end_gen=int(end_gen),
                skip_init=bool(skip_init),
            )
            if _cuda_architecture_b_result is not None:
                _slot_terminal_status = "complete"
                return _cuda_architecture_b_result

            for gen in range(int(start_gen), int(end_gen)):
                gen_t0 = time.perf_counter()
                try:
                    setattr(self, "_current_gen", int(gen))
                    setattr(self.pop, "_eiel_current_generation_index", int(gen))
                    setattr(self.pop, "_current_generation", int(gen))
                except Exception:
                    pass

                # v2-only checkpointing: track whether a bundle was already saved
                # at this generation SafeBoundary (periodic vs control-plane vs interrupt).
                bundle_v2_saved_this_boundary = False
                bundle_v2_saved_path = None

                # Phase 3 determinism: install trace context for membership write localization.
                try:
                    from simulation.membership_trace import set_trace_context  # noqa: WPS433
                    set_trace_context(gen=int(gen), phase="loop.pre_start_generation")
                except Exception:
                    pass
                # Phase 3 determinism probe: RNG state hashes at generation entry (no RNG draws).
                try:
                    from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                    _maybe_write_probe(self, generation_index=int(gen), tag="loop.gen_entry")
                except Exception:
                    pass


                # ------------------------------------------------------------------
                # Phase 3 determinism: canonicalize lazy transient state BEFORE the
                # pre_start_generation_alive probe.
                #
                # Resume path materializes certain transient buffers (drive arrays,
                # GRU bank/hidden memory) during checkpoint apply. Fresh path can
                # leave them uninitialized until later, which makes the earliest
                # probe boundary non-comparable (None vs dict, sha mismatches).
                #
                # This sync must be:
                #   - idempotent
                #   - deterministic
                #   - RNG-free
                # ------------------------------------------------------------------
                try:
                    sync_fn = getattr(self, "_phase3_pre_start_generation_sync", None)
                    if callable(sync_fn):
                        alive_agents = list(self.pop.alive_agents())
                        sync_fn(alive_agents)
                except Exception:
                    pass

                # Phase 3 probe: snapshot membership BEFORE generation lifecycle.
                _maybe_dump_alive_probe(generation_index=int(gen), tag="pre_start_generation_alive")

                t_start_generation0 = _perf_time(self)
                (
                    buckets,
                    shock,
                    injury_recover_mult,
                    deaths_this_gen,
                    successful_births,
                    alive_start,
                    alive_ids,
                    alive_id_set,
                    cohort_size,
                    gen_peak_injury,
                    failed_births,
                    rest_streaks,
                ) = self._start_generation(gen)
                _perf_add(self, buckets, "gen_start_total", t_start_generation0)

                _cxxrewrite1_lifecycle_requested = (
                    bool(getattr(self.sim_cfg, "cxxrewrite_array_backend_enable", False))
                    or bool(getattr(self.sim_cfg, "cxxrewrite_lifecycle_shadow_enable", False))
                    or bool(getattr(self.sim_cfg, "cxxrewrite_lifecycle_authoritative_enable", False))
                    or bool(getattr(self.sim_cfg, "cxxrewrite_lifecycle_record_churn_tape_enable", False))
                    or bool(getattr(self.sim_cfg, "cxxrewrite_lifecycle_validate_shadow_enable", False))
                )
                _cxxrewrite1_lifecycle_start_state = None

                try:
                    _array_backend_requested = (
                        bool(getattr(self.sim_cfg, "runner_array_backend_enable", False))
                        or bool(getattr(self.sim_cfg, "cxxrewrite_array_backend_shadow_enable", False))
                        or bool(getattr(self.sim_cfg, "cxxrewrite_array_backend_authoritative_enable", False))
                        or bool(getattr(self.sim_cfg, "cxxrewrite_array_backend_debug_sync_enable", False))
                    )
                    if _array_backend_requested:
                        from perf.cxxrewrite0_array_backend import record_generation_shadow  # noqa: WPS433

                        t_array_backend0 = _perf_time(self)
                        record_generation_shadow(
                            self,
                            alive=alive_start,
                            generation=int(gen),
                            phase="start_generation",
                            buckets=buckets,
                        )
                        _perf_add(self, buckets, "cxxrewrite0_array_backend_total", t_array_backend0)
                except Exception:
                    if bool(getattr(self.sim_cfg, "cxxrewrite_array_backend_fail_fast", False)) or bool(
                        getattr(self.sim_cfg, "cxxrewrite_array_backend_authoritative_enable", False)
                    ):
                        raise

                try:
                    if _cxxrewrite1_lifecycle_requested:
                        from perf.cxxrewrite1_array_lifecycle import begin_generation_lifecycle  # noqa: WPS433

                        t_cxxrewrite1_lifecycle0 = _perf_time(self)
                        _cxxrewrite1_lifecycle_start_state = begin_generation_lifecycle(
                            self,
                            alive=alive_start,
                            generation=int(gen),
                            buckets=buckets,
                        )
                        _perf_add(self, buckets, "cxxrewrite1_lifecycle_begin_total", t_cxxrewrite1_lifecycle0)
                except Exception:
                    if bool(getattr(self.sim_cfg, "cxxrewrite_lifecycle_fail_closed", False)) or bool(
                        getattr(self.sim_cfg, "cxxrewrite_lifecycle_authoritative_enable", False)
                    ):
                        raise

                t_run_steps0 = _perf_time(self)
                _flat_output_cost_timing = getattr(self, "_flat_output_cost_timing", None)
                if _flat_output_cost_timing is None:
                    deaths_this_gen, successful_births, failed_births, gen_peak_injury, steps_ran = self._run_steps(
                        gen=gen,
                        buckets=buckets,
                        injury_recover_mult=injury_recover_mult,
                        deaths_this_gen=deaths_this_gen,
                        successful_births=successful_births,
                        failed_births=failed_births,
                        gen_peak_injury=gen_peak_injury,
                        rest_streaks=rest_streaks,
                    )
                else:
                    with _flat_output_cost_timing.simulation_phase():
                        deaths_this_gen, successful_births, failed_births, gen_peak_injury, steps_ran = self._run_steps(
                            gen=gen,
                            buckets=buckets,
                            injury_recover_mult=injury_recover_mult,
                            deaths_this_gen=deaths_this_gen,
                            successful_births=successful_births,
                            failed_births=failed_births,
                            gen_peak_injury=gen_peak_injury,
                            rest_streaks=rest_streaks,
                        )
                _perf_add(self, buckets, "tick_loop_total", t_run_steps0)

                alive_after = list(self.pop.alive_agents())
                alive_end_ids = {a.agent_id for a in alive_after}

                # Phase 3 determinism probe: state AFTER run_steps (before selection/metrics).
                try:
                    from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                    _maybe_write_probe(self, generation_index=int(gen), tag="loop.post_run_steps")
                except Exception:
                    pass


                # Phase 3 probe: snapshot membership AFTER run_steps (before selection/metrics).
                _maybe_dump_alive_probe(generation_index=int(gen), tag="post_run_steps_alive")


                cohort_survivors = len(alive_id_set.intersection(alive_end_ids))
                cohort_survival_rate = (cohort_survivors / cohort_size) if cohort_size > 0 else 0.0

                if gen < 2:
                    print(
                        f"[gen {gen}] cohort_size(start)={cohort_size} "
                        f"alive_end(before_cull)={len(alive_end_ids)} "
                        f"cohort_survivors={cohort_survivors} "
                        f"cohort_survival_rate={cohort_survival_rate:.3f} "
                        f"shock={int(shock)}"
                    )

                # ---- FIX: selection must run BEFORE metrics so parent snapshots exist ----
                t_selection0 = _perf_time(self)
                _eligible_ids, _selected_parent_ids = self._selection_and_behaviors(
                    gen=gen,
                    alive_after=alive_after,
                    alive_id_set=alive_id_set,
                    alive_end_ids=alive_end_ids,
                    buckets=buckets,
                )
                _perf_add(self, buckets, "selection_total", t_selection0)
                if _lineage_funnel_enabled(getattr(self, "sim_cfg", None)):
                    self._lineage_reproduction_funnel_pre_turnover_gen = _capture_lineage_funnel_pre_turnover(
                        generation=int(gen),
                        alive_start_agents=alive_start,
                        alive_after_agents=alive_after,
                        eligible_ids=_eligible_ids,
                        selected_parent_agents=list(getattr(self, "_selected_parent_agents_gen", []) or []),
                        successful_births=int(successful_births),
                        failed_births=int(failed_births),
                        birth_failure_counts={
                            "birth_fail_no_space": int(getattr(self, "_birth_fail_no_space_gen", 0)),
                            "birth_fail_no_mate": int(getattr(self, "_birth_fail_no_mate_gen", 0)),
                            "birth_fail_buffer": int(getattr(self, "_birth_fail_buffer_gen", 0)),
                            "birth_fail_injury_gate": int(getattr(self, "_birth_fail_injury_gate_gen", 0)),
                            "birth_fail_scar_gate": int(getattr(self, "_birth_fail_scar_gate_gen", 0)),
                            "birth_fail_conception": int(getattr(self, "_birth_fail_conception_gen", 0)),
                            "birth_fail_gestation": int(getattr(self, "_birth_fail_gestation_gen", 0)),
                        },
                        birth_contexts=list(getattr(self, "_lineage_reproduction_funnel_birth_contexts_gen", []) or []),
                        sim_cfg=getattr(self, "sim_cfg", None),
                        env_cfg=getattr(self, "env_cfg", None),
                        birth_event_contexts=list(getattr(self, "_lineage_reproduction_funnel_event_contexts_gen", []) or []),
                        scan_action_event_contexts=list(
                            getattr(self, "_lineage_reproduction_funnel_scan_action_contexts_gen", []) or []
                        ),
                        parent_fraction=float(getattr(getattr(self, "pop_cfg", None), "parent_fraction", 0.0)),
                    )
                if _structural_selection_funnel_enabled(getattr(self, "sim_cfg", None)):
                    self._structural_selection_funnel_pre_turnover_gen = _capture_structural_selection_funnel_pre_turnover(
                        self,
                        generation=int(gen),
                        alive_start_agents=alive_start,
                        alive_after_agents=alive_after,
                        eligible_ids=_eligible_ids,
                        selected_parent_agents=list(getattr(self, "_selected_parent_agents_gen", []) or []),
                        birth_contexts=list(getattr(self, "_lineage_reproduction_funnel_birth_contexts_gen", []) or []),
                    )

                # Phase 3 determinism probe: state AFTER selection (before metrics finalize).
                try:
                    from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                    _maybe_write_probe(self, generation_index=int(gen), tag="loop.post_selection")
                except Exception:
                    pass

                # Now metrics can safely read self._selected_parent_agents_gen
                t_metrics0 = _perf_time(self)
                self._finish_generation_metrics(
                    gen=gen,
                    alive_after=alive_after,
                    deaths_this_gen=deaths_this_gen,
                    successful_births=successful_births,
                    failed_births=failed_births,
                    cohort_size=cohort_size,
                    cohort_survivors=cohort_survivors,
                    cohort_survival_rate=cohort_survival_rate,
                    gen_peak_injury=gen_peak_injury,
                    shock=shock,
                    log_gen_params=log_gen_params,
                    buckets=buckets,
                )
                _perf_add(self, buckets, "metrics_total", t_metrics0)
                try:
                    buckets["csv_write_total"] = float(buckets.get("csv_write_total", 0.0)) + float(
                        getattr(self.logger, "_last_perf_csv_write_s", 0.0) or 0.0
                    )
                    buckets["lineage_metrics_write_total"] = float(
                        buckets.get("lineage_metrics_write_total", 0.0)
                    ) + float(getattr(self.logger, "_last_perf_lineage_write_s", 0.0) or 0.0)
                except Exception:
                    pass
                try:
                    _maw = getattr(self, "memory_action_window_instrumentation", None)
                    if _maw is not None:
                        _maw.flush_generation_summary(self, generation=int(gen))
                except Exception:
                    raise
                try:
                    _mm4 = getattr(self, "mem_max_live_level4_instrumentation", None)
                    if _mm4 is not None:
                        _mm4.flush_generation_summary(self, generation=int(gen))
                except Exception:
                    raise
                try:
                    _pdac = getattr(self, "prediction_demand_action_context_instrumentation", None)
                    if _pdac is not None:
                        _pdac.flush_generation_summary(self, generation=int(gen))
                except Exception:
                    raise
                try:
                    from simulation.corridor_cp0_fixed_probe import write_generation_summary as _cp0_probe_summary
                    _cp0_probe_summary(self, generation=int(gen))
                except Exception:
                    if bool(getattr(getattr(self, "sim_cfg", None), "corridor_cp0_telemetry_enable", False)):
                        raise

                try:
                    from simulation.membership_trace import set_trace_context  # noqa: WPS433
                    set_trace_context(gen=int(gen), phase="loop.post_metrics.reproduce_and_cull")
                except Exception:
                    pass

                # Phase 3 determinism probe: state BEFORE reproduce_and_cull (end-of-gen, pre-repro).
                try:
                    from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                    _maybe_write_probe(self, generation_index=int(gen), tag="loop.gen_exit_pre_reproduce")
                except Exception:
                    pass


                # -----------------------------
                # Phase 5B: Trait Injection hook plumbing
                # -----------------------------
                # Population.reproduce_and_cull() includes an optional trait injection hook.
                # That hook is generation-indexed and needs the run directory to locate the
                # ControlPlane event log (<run_dir>/events/command_events.jsonl).
                #
                # Without these attributes, TraitInjectionCommand interventions are accepted
                # and event-logged, but never actually applied to genomes.
                try:
                    setattr(self.pop, "_eiel_current_generation_index", int(gen))
                    setattr(self.pop, "_current_generation", int(gen))
                    if not str(getattr(self.pop, "run_dir", "") or ""):
                        setattr(self.pop, "run_dir", str(getattr(self, "run_dir", "") or ""))
                except Exception:
                    pass

                from native_runtime.safe_default import native_generation_requested  # noqa: WPS433

                native_runtime_active = bool(native_generation_requested(self))
                native_cxx26_scope_lock = bool(
                    native_runtime_active
                    and bool(getattr(self.sim_cfg, "native_runtime_cxx26_scope_lock_enable", False))
                )
                t_generation_repro0 = _perf_time(self)
                if native_cxx26_scope_lock and int(getattr(self.sim_cfg, "generations", 1) or 1) > 1:
                    from native_runtime.runner_integration import (  # noqa: WPS433
                        execute_native_generation_boundary_turnover_from_runner,
                    )

                    execute_native_generation_boundary_turnover_from_runner(
                        self,
                        generation_index=int(gen),
                    )
                elif not native_cxx26_scope_lock:
                    self.pop.reproduce_and_cull()
                _perf_add(self, buckets, "generation_reproduce_cull_total", t_generation_repro0)
                if bool(getattr(self.sim_cfg, "recomb72_audit_enable", False)):
                    from evolution.recomb72_live_fixture import record_recomb72_live_generation_v1

                    record_recomb72_live_generation_v1(
                        self,
                        generation=int(gen) + 1,
                    )
                if (not native_cxx26_scope_lock) and _lineage_funnel_enabled(getattr(self, "sim_cfg", None)):
                    _write_lineage_funnel_generation(
                        run_dir=str(getattr(self, "run_dir", "") or ""),
                        sim_cfg=getattr(self, "sim_cfg", None),
                        generation=int(gen),
                        pre_turnover_snapshot=dict(getattr(self, "_lineage_reproduction_funnel_pre_turnover_gen", {}) or {}),
                        post_refill_agents=list(self.pop.alive_agents()),
                        turnover_snapshot=dict(getattr(self.pop, "_lineage_reproduction_funnel_turnover_gen", {}) or {}),
                    )
                if (not native_cxx26_scope_lock) and _structural_selection_funnel_enabled(getattr(self, "sim_cfg", None)):
                    _write_structural_selection_funnel_generation(
                        runner=self,
                        run_dir=str(getattr(self, "run_dir", "") or ""),
                        generation=int(gen),
                        pre_turnover_snapshot=dict(getattr(self, "_structural_selection_funnel_pre_turnover_gen", {}) or {}),
                        post_refill_agents=list(self.pop.alive_agents()),
                        turnover_snapshot=dict(getattr(self.pop, "_structural_selection_funnel_turnover_gen", {}) or {}),
                    )
                if (not native_cxx26_scope_lock) and _structural_tick_reproduction_funnel_enabled(getattr(self, "sim_cfg", None)):
                    _write_structural_tick_reproduction_funnel_generation(
                        runner=self,
                        run_dir=str(getattr(self, "run_dir", "") or ""),
                        generation=int(gen),
                        alive_after_agents=list(alive_after or []),
                        scan_action_event_contexts=list(
                            getattr(self, "_lineage_reproduction_funnel_scan_action_contexts_gen", []) or []
                        ),
                        birth_event_contexts=list(getattr(self, "_lineage_reproduction_funnel_event_contexts_gen", []) or []),
                        birth_contexts=list(getattr(self, "_lineage_reproduction_funnel_birth_contexts_gen", []) or []),
                        post_refill_agents=list(self.pop.alive_agents()),
                    )
                if hasattr(self, "_mark_bank_dirty"):
                    self._mark_bank_dirty()


                # Phase 3 determinism probe: state AFTER reproduce_and_cull, for next generation entry.
                # This narrows drift at gen boundaries (genN->genN+1).
                try:
                    from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                    _maybe_write_probe(self, generation_index=int(gen) + 1, tag="loop.gen_transition_post_reproduce")
                except Exception:
                    pass

                try:
                    from simulation.membership_trace import set_trace_context  # noqa: WPS433
                    set_trace_context(gen=int(gen), phase="loop.post_repro.envsync")
                except Exception:
                    pass
                t_envsync0 = _perf_time(self)
                self._sync_env_with_population()
                _perf_add(self, buckets, "env_population_sync_total", t_envsync0)

                t_typed_lifecycle_output0 = _perf_time(self)
                _emit_route_neutral_typed_lifecycle_output(self)
                _perf_add(self, buckets, "typed_lifecycle_public_output_total", t_typed_lifecycle_output0)

                try:
                    if _cxxrewrite1_lifecycle_requested:
                        from perf.cxxrewrite1_array_lifecycle import finish_generation_lifecycle  # noqa: WPS433

                        t_cxxrewrite1_lifecycle_finish0 = _perf_time(self)
                        finish_generation_lifecycle(
                            self,
                            start_state=_cxxrewrite1_lifecycle_start_state,
                            generation=int(gen),
                            phase="post_envsync",
                            buckets=buckets,
                        )
                        _perf_add(self, buckets, "cxxrewrite1_lifecycle_finish_total", t_cxxrewrite1_lifecycle_finish0)
                except Exception:
                    if bool(getattr(self.sim_cfg, "cxxrewrite_lifecycle_fail_closed", False)) or bool(
                        getattr(self.sim_cfg, "cxxrewrite_lifecycle_authoritative_enable", False)
                    ):
                        raise

                # -----------------------------
                # SafeBoundary step (6): ControlPlane.process_pending (Phase 2)
                # -----------------------------
                # SB-ORDER-01 / REPLAY-ORDER-01: do not reorder the boundary.
                # We capture RNG snapshot pre-command (best-effort) and then apply
                # pending commands at the boundary (no timestamps / no RNG draws).
                try:
                    t_control_plane0 = _perf_time(self)
                    next_gen = int(gen) + 1
                    try:
                        from simulation.membership_trace import set_trace_context  # noqa: WPS433
                        set_trace_context(gen=int(next_gen), phase="safeboundary.control_plane")
                    except Exception:
                        pass
                    # (4) rng_snapshot_capture(next_gen) [pre-command]
                    try:
                        snap = None
                        if getattr(self, "rng_manager", None) is not None:
                            snap = self.rng_manager.capture_snapshot(generation_index=next_gen)
                        if snap is not None:
                            setattr(self, "_last_rng_snapshot_v2", snap)
                            setattr(self, "_last_safe_boundary_next_gen", next_gen)
                    except Exception:
                        pass
                    # (5) SafeBoundaryContext_create(next_gen) [minimal dict for Phase 2]
                    sb_ctx = {
                        "schema_version": "safe_boundary_context_v1",
                        "generation_index": next_gen,
                        "event_log_length": None,
                    }
                    # Phase D capacity epochs are integrity-only and execute at
                    # this completed-generation boundary before any next-
                    # generation RNG draw.  The manager must already be
                    # explicitly authorized; it cannot authorize itself.
                    capacity_manager = getattr(self, "substrate_capacity_manager", None)
                    if capacity_manager is not None:
                        from simulation.capacity_epoch import (
                            CapacityEpochState,
                            SAFE_BOUNDARY_KIND,
                            capacity_epoch_rng_digest,
                        )

                        if capacity_manager.state in {
                            CapacityEpochState.MONITORING,
                            CapacityEpochState.WARNING,
                            CapacityEpochState.ELIGIBLE,
                        }:
                            try:
                                capacity_state, pressure_sample = capacity_manager.observe_agents(
                                    list(getattr(self.pop, "agents", {}).values()),
                                    generation=int(gen),
                                )
                                self.sim_cfg.capacity_epoch_transition_state = capacity_state.value
                                setattr(self, "_last_capacity_pressure_sample", pressure_sample)
                                setattr(self, "_last_capacity_pressure_state", capacity_state.value)
                            except Exception as exc:
                                raise _CapacityEpochMonitoringFailure(
                                    "capacity epoch monitoring failed closed"
                                ) from exc

                        if capacity_manager.state == CapacityEpochState.AUTHORIZED:
                            try:
                                rollback_root = os.path.join(self.run_dir, "capacity_epoch_rollbacks")
                                os.makedirs(rollback_root, exist_ok=True)
                                rollback_path = os.path.join(rollback_root, f"bundle_gen{next_gen:06d}")

                                def _write_capacity_rollback() -> str:
                                    checkpoint_bundle_v2_save(
                                        self,
                                        rollback_path,
                                        generation_index=int(next_gen),
                                        strict_replay_mode=True,
                                    )
                                    return rollback_path

                                def _verify_capacity_bundle(path: str) -> bool:
                                    from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_load

                                    loaded = checkpoint_bundle_v2_load(path)
                                    return isinstance(loaded, dict) and isinstance(loaded.get("metadata"), dict)

                                rng_before = self.rng_manager.capture_snapshot(generation_index=next_gen)
                                receipt = capacity_manager.migrate_at_safe_boundary(
                                    agents=list(getattr(self.pop, "agents", {}).values()),
                                    sim_cfg=self.sim_cfg,
                                    next_generation=next_gen,
                                    boundary_kind=SAFE_BOUNDARY_KIND,
                                    write_rollback_checkpoint=_write_capacity_rollback,
                                    verify_rollback_checkpoint=_verify_capacity_bundle,
                                    rng_digest_before=capacity_epoch_rng_digest(rng_before),
                                    rng_digest_after=lambda: capacity_epoch_rng_digest(
                                        self.rng_manager.capture_snapshot(generation_index=next_gen)
                                    ),
                                )
                                migrated_root = os.path.join(self.run_dir, "capacity_epoch_migrated")
                                os.makedirs(migrated_root, exist_ok=True)
                                migrated_path = os.path.join(migrated_root, f"bundle_gen{next_gen:06d}")
                                checkpoint_bundle_v2_save(
                                    self,
                                    migrated_path,
                                    generation_index=int(next_gen),
                                    strict_replay_mode=True,
                                )
                                if not _verify_capacity_bundle(migrated_path):
                                    raise RuntimeError("capacity migrated checkpoint verification failed")
                                receipt["migrated_checkpoint_path"] = migrated_path
                                setattr(self, "_capacity_epoch_transition_receipt", receipt)
                            except Exception as exc:
                                capacity_manager.state = CapacityEpochState.FAILED_ROLLBACK
                                raise _CapacityEpochMigrationFailure(
                                    "capacity epoch migration failed closed"
                                ) from exc
                    # (6) ControlPlane.process_pending(next_gen)
                    cp = getattr(self, "control_plane", None)
                    if cp is not None:
                        applied = cp.process_pending(next_gen, sb_ctx, sim=self)
                        # -----------------------------
                        # Phase 5d: UI timeline (best-effort, non-determinism-critical)
                        # -----------------------------
                        # The WebUI expects viz/applied_commands_log.jsonl to exist and to
                        # contain a basic timeline of applied commands.
                        #
                        # ControlPlane itself is determinism-critical and must not do any
                        # best-effort UI writes, so we append here in the runner.
                        try:
                            vw = getattr(self, "viz", None)
                            if vw is not None and bool(getattr(vw, "enabled", False)):
                                for _ev in (applied or []):
                                    try:
                                        _cu = str(getattr(_ev, "command_uuid", "") or "")
                                    except Exception:
                                        _cu = ""

                                    _payload = {}
                                    _ph = ""
                                    try:
                                        _info = getattr(cp, "_known", {}).get(_cu, None)
                                        if isinstance(_info, dict):
                                            _payload = _info.get("payload", {}) or {}
                                            _ph = str(_info.get("payload_hash", "") or "")
                                    except Exception:
                                        _payload = {}
                                        _ph = ""

                                    _ctype = str(_payload.get("command_type", "") or "")
                                    _ap = ""
                                    if _ctype == "TraitInjectionCommand":
                                        _ti = _payload.get("trait_injection", {}) or {}
                                        if isinstance(_ti, dict):
                                            _ap = str(_ti.get("apply_phase", "") or "")

                                    try:
                                        vw.append_applied_command_event(
                                            command_uuid=_cu,
                                            command_type=_ctype,
                                            apply_phase=_ap,
                                            applied_generation=int(next_gen),
                                            payload_hash=(_ph or str(getattr(_ev, "command_payload_hash", "") or "")),
                                            ack_status="APPLIED",
                                            error_reason=None,
                                        )
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                        # Keep runner mirror for checkpoint metadata persistence.
                        try:
                            setattr(self, "_control_plane_submission_sequence_number", int(cp.get_submission_sequence_number()))
                        except Exception:
                            pass
                        # CheckpointRequestCommand: optional immediate checkpoint (v2-only)
                        if bool(getattr(self, "_control_plane_checkpoint_now", False)):
                            try:
                                rotate = int(getattr(self.sim_cfg, "checkpoint_rotate", 0) or 0)

                                # Prefer v2 subdir; fall back to legacy checkpoint_subdir if provided.
                                v2_subdir = str(getattr(self.sim_cfg, "checkpoint_bundle_v2_subdir", "") or "").strip()
                                if not v2_subdir:
                                    v2_subdir = str(getattr(self.sim_cfg, "checkpoint_subdir", "checkpoint_bundles_v2") or "checkpoint_bundles_v2")
                                v2_strict = bool(getattr(self.sim_cfg, "checkpoint_bundle_v2_strict_replay_mode", True))

                                root = os.path.join(self.run_dir, v2_subdir)
                                os.makedirs(root, exist_ok=True)

                                setattr(self, "_current_gen", int(gen))
                                bpath = os.path.join(root, f"bundle_gen{int(next_gen):06d}")

                                # Only save once per SafeBoundary.
                                if not bool(bundle_v2_saved_this_boundary):
                                    t_checkpoint_now0 = _perf_time(self)
                                    checkpoint_bundle_v2_save(self, bpath, generation_index=int(next_gen), strict_replay_mode=v2_strict)
                                    _perf_add_checkpoint_details(self, buckets)
                                    rotate_checkpoint_bundles_v2(root, keep_last=rotate)
                                    checkpoint_elapsed = _perf_elapsed(self, t_checkpoint_now0)
                                    buckets["checkpoint_total"] = float(buckets.get("checkpoint_total", 0.0)) + checkpoint_elapsed
                                    buckets["checkpoint_control_plane_total"] = float(
                                        buckets.get("checkpoint_control_plane_total", 0.0)
                                    ) + checkpoint_elapsed
                                    bundle_v2_saved_this_boundary = True
                                    bundle_v2_saved_path = str(bpath)
                            except Exception as _control_checkpoint_exc:
                                _raise_native_checkpoint_failure(
                                    self,
                                    _control_checkpoint_exc,
                                    stage="checkpoint_control_plane",
                                )
                                pass
                            try:
                                setattr(self, "_control_plane_checkpoint_now", False)
                            except Exception:
                                pass
                except _NativeActiveCheckpointFailure:
                    raise
                except _CapacityEpochMigrationFailure:
                    raise
                except _CapacityEpochMonitoringFailure:
                    raise
                except Exception:
                    # ControlPlane must never crash the sim loop.
                    pass
                else:
                    _perf_add(self, buckets, "control_plane_total", t_control_plane0)


                # -----------------------------
                # Phase 5a: MetricsEngine.compute_post_boundary (EPI + AV)
                # -----------------------------
                # Must run post-command at the SafeBoundary without mutating main sim state.
                try:
                    t_post_boundary_metrics0 = _perf_time(self)
                    next_gen = int(gen) + 1
                    if bool(getattr(self.sim_cfg, "epi_enable", False)):
                        me = getattr(self, "_phase5_metrics_engine", None)
                        if me is None:
                            from simulation.metrics_pkg.metrics_engine import Phase5MetricsEngine  # noqa: WPS433
                            me = Phase5MetricsEngine(run_dir=self.run_dir)
                            setattr(self, "_phase5_metrics_engine", me)
                        me.compute_post_boundary(sim=self, generation_index=next_gen)
                    _perf_add(self, buckets, "post_boundary_metrics_total", t_post_boundary_metrics0)
                except Exception as _epi_exc:
                    # Metrics must never crash the sim loop.
                    # But if EPI is enabled and it fails, we emit a visible
                    # breadcrumb to console for validation runs.
                    try:
                        if bool(getattr(self.sim_cfg, "epi_enable", False)):
                            print(f"[phase5a] WARNING: EPI/AV failed at boundary gen {int(gen)+1}: {_epi_exc}", flush=True)
                    except Exception:
                        pass
                    pass

                # -----------------------------
                # Phase E1: deterministic evolvability metrics
                # -----------------------------
                try:
                    t_phase_e_metrics0 = _perf_time(self)
                    next_gen = int(gen) + 1
                    if bool(getattr(self.sim_cfg, "phase_e_e1_enable", False)):
                        e1 = getattr(self, "_phase_e_metrics_engine", None)
                        if e1 is None:
                            from simulation.metrics_pkg.phase_e_metrics_engine import PhaseEMetricsEngine  # noqa: WPS433

                            e1 = PhaseEMetricsEngine(
                                run_dir=self.run_dir,
                                retention_k=max(1, int(getattr(self.sim_cfg, "phase_e_e1_retention_k_gen", 3) or 3)),
                                neutrality_window=max(2, int(getattr(self.sim_cfg, "phase_e_e1_neutrality_window", 8) or 8)),
                            )
                            setattr(self, "_phase_e_metrics_engine", e1)
                        e1.compute_post_boundary(sim=self, generation_index=next_gen)
                    _perf_add(self, buckets, "phase_e_metrics_total", t_phase_e_metrics0)
                except Exception as _e1_exc:
                    try:
                        if bool(getattr(self.sim_cfg, "phase_e_e1_enable", False)):
                            print(f"[phase_e] WARNING: E1 metrics failed at boundary gen {int(gen)+1}: {_e1_exc}", flush=True)
                    except Exception:
                        pass
                    pass

                # -----------------------------
                # Phase E2: deterministic failure taxonomy
                # -----------------------------
                try:
                    t_phase_e_taxonomy0 = _perf_time(self)
                    next_gen = int(gen) + 1
                    if bool(getattr(self.sim_cfg, "phase_e_e2_enable", False)):
                        e2 = getattr(self, "_phase_e_failure_taxonomy_engine", None)
                        if e2 is None:
                            from simulation.metrics_pkg.phase_e_metrics_engine import PhaseEFailureTaxonomyEngine  # noqa: WPS433

                            e2 = PhaseEFailureTaxonomyEngine(run_dir=self.run_dir)
                            setattr(self, "_phase_e_failure_taxonomy_engine", e2)
                        e2.compute_post_boundary(sim=self, generation_index=next_gen)
                    _perf_add(self, buckets, "phase_e_taxonomy_total", t_phase_e_taxonomy0)
                except Exception as _e2_exc:
                    try:
                        if bool(getattr(self.sim_cfg, "phase_e_e2_enable", False)):
                            print(f"[phase_e] WARNING: E2 taxonomy failed at boundary gen {int(gen)+1}: {_e2_exc}", flush=True)
                    except Exception:
                        pass
                    pass


                # -----------------------------
                # Phase 6 (draft): Pressure Autopilot (post-metrics, pre-hash)
                # -----------------------------
                # Runs after EPI/AV is computed (so it can use those signals), and
                # before SafeBoundary hash logging (so WAL/plan updates are captured).
                try:
                    t_pressure_autopilot0 = _perf_time(self)
                    next_gen = int(gen) + 1
                    ap = getattr(self, "pressure_autopilot", None)
                    if ap is not None:
                        ap.maybe_tick(sim=self, generation_index=next_gen)
                    _perf_add(self, buckets, "pressure_autopilot_total", t_pressure_autopilot0)
                except Exception:
                    # Autopilot must never crash the sim loop.
                    pass


                # -----------------------------
                # Phase 3: SafeBoundary hash log + Replay hook
                # -----------------------------
                try:
                    t_boundary_hash0 = _perf_time(self)
                    next_gen = int(gen) + 1
                    if bool(getattr(self.sim_cfg, "safeboundary_hash_log_enable", False)):
                        hashes = compute_safeboundary_hashes(self)
                        append_safeboundary_hash_log(self.run_dir, next_gen, hashes)
                    # Replay hook (installed by simulation/replay_engine.py)
                    rh = getattr(self, "_replay_hook", None)
                    if callable(rh):
                        rh(next_gen)
                    _perf_add(self, buckets, "boundary_hash_replay_total", t_boundary_hash0)
                except Exception:
                    # Never crash the sim loop on hash tooling.
                    pass

                # -----------------------------
                # Phase 3: Optional boundary-state dumps (targeted)
                # -----------------------------
                # Determinism debug aid. Disabled unless:
                #   EIEL_STATE_DUMP=1 and EIEL_STATE_DUMP_GENS contains next_gen.
                # Kept narrow on purpose (no "logging everywhere").
                try:
                    if bool(int(os.environ.get("EIEL_STATE_DUMP", "0") or "0")):
                        gens = os.environ.get("EIEL_STATE_DUMP_GENS", "").strip()
                        if gens:
                            wanted = {int(x) for x in gens.split(",") if x.strip().isdigit()}
                            next_gen = int(gen) + 1
                            if next_gen in wanted:
                                try:
                                    from simulation.state_dump import maybe_dump_state  # type: ignore
                                except Exception:
                                    from engine.simulation.state_dump import maybe_dump_state  # type: ignore  # noqa: WPS433

                                maybe_dump_state(self, generation_index=next_gen, tag="safeboundary_post")
                except Exception:
                    pass

                # -----------------------------
                # Phase 3: Periodic checkpointing (Checkpoint Bundle v2)
                # -----------------------------
                try:
                    rotate = int(getattr(self.sim_cfg, "checkpoint_rotate", 0) or 0)

                    from native_runtime.safe_default import qualified_checkpoint_period_gen  # noqa: WPS433

                    v2_period = qualified_checkpoint_period_gen(self.sim_cfg, self)

                    # Prefer v2 subdir; fall back to legacy checkpoint_subdir if provided.
                    v2_subdir = str(getattr(self.sim_cfg, "checkpoint_bundle_v2_subdir", "") or "").strip()
                    if not v2_subdir:
                        v2_subdir = str(getattr(self.sim_cfg, "checkpoint_subdir", "checkpoint_bundles_v2") or "checkpoint_bundles_v2")

                    v2_strict = bool(getattr(self.sim_cfg, "checkpoint_bundle_v2_strict_replay_mode", True))

                    if v2_period > 0:
                        next_gen = int(gen) + 1
                        if (next_gen % v2_period) == 0 and (not bool(bundle_v2_saved_this_boundary)):
                            root = os.path.join(self.run_dir, v2_subdir)
                            os.makedirs(root, exist_ok=True)
                            bpath = os.path.join(root, f"bundle_gen{next_gen:06d}")
                            t_checkpoint_periodic0 = _perf_time(self)
                            checkpoint_bundle_v2_save(self, bpath, generation_index=int(next_gen), strict_replay_mode=v2_strict)
                            _perf_add_checkpoint_details(self, buckets)
                            rotate_checkpoint_bundles_v2(root, keep_last=rotate)
                            checkpoint_elapsed = _perf_elapsed(self, t_checkpoint_periodic0)
                            buckets["checkpoint_total"] = float(buckets.get("checkpoint_total", 0.0)) + checkpoint_elapsed
                            buckets["checkpoint_periodic_total"] = float(
                                buckets.get("checkpoint_periodic_total", 0.0)
                            ) + checkpoint_elapsed
                            bundle_v2_saved_this_boundary = True
                            bundle_v2_saved_path = str(bpath)
                except Exception as _v2_exc:
                    _raise_native_checkpoint_failure(
                        self,
                        _v2_exc,
                        stage="checkpoint_periodic_generation_boundary",
                    )
                    try:
                        print(f"[checkpoint_v2] WARNING: failed to save bundle at boundary gen {int(gen)+1}: {_v2_exc}")
                    except Exception:
                        pass

                t_behavior_end0 = _perf_time(self)
                self.behavior_logger.end_generation()
                _perf_add(self, buckets, "behavior_logging_total", t_behavior_end0)

                t_progress0 = _perf_time(self)
                progress.log(gen)
                _perf_add(self, buckets, "progress_logging_total", t_progress0)
                if _slot_lease is not None:
                    try:
                        _slot_lease.update(
                            current_generation=int(gen) + 1,
                            status="running",
                            stage=f"generation {int(gen) + 1}",
                        )
                    except Exception:
                        pass

                # WebUI liveness heartbeat (operational; safe to ignore failures).
                t_heartbeat0 = _perf_time(self)
                _write_heartbeat(self.run_dir, int(gen))
                _perf_add(self, buckets, "heartbeat_write_total", t_heartbeat0)

                # -----------------------------
                # Checkpoint v1 writer removed (v2-only engine)
                # -----------------------------


                # -----------------------------
                # Phase 5d: viz/ per-generation summaries + run status (observability only)
                # -----------------------------
                try:
                    t_viz0 = _perf_time(self)
                    _viz_last_completed_gen = int(gen)
                    if _viz is not None and getattr(_viz, "enabled", False):
                        viz_alive = int(len(alive_after))
                        native_public_counts = getattr(self, "_native_runtime_public_output_counts", None)
                        if isinstance(native_public_counts, dict) and "alive" in native_public_counts:
                            viz_alive = int(native_public_counts["alive"])
                        _viz.append_metrics_summary(
                            generation=int(gen),
                            alive=viz_alive,
                            births=int(successful_births),
                            deaths=int(deaths_this_gen),
                            avg_gen_time_s=float(time.perf_counter() - gen_t0),
                        )
                        _viz.write_run_status(
                            status="running",
                            current_generation=int(gen) + 1,
                            target_generation=int(end_gen),
                            last_error=None,
                            branch_state="RUNNING",
                            run_kind=str(_viz_run_kind),
                        )
                    _perf_add(self, buckets, "viz_write_total", t_viz0)
                except Exception:
                    pass


                # -----------------------------
                # Graceful interrupt checkpoint (Ctrl-C / SIGINT)
                # -----------------------------
                if getattr(self, "_stop_requested", False):
                    # v2-only: ensure a Checkpoint Bundle v2 exists at the next SafeBoundary, then exit.
                    try:
                        rotate = int(getattr(self.sim_cfg, "checkpoint_rotate", 0) or 0)

                        # Prefer v2 subdir; fall back to legacy checkpoint_subdir if provided.
                        v2_subdir = str(getattr(self.sim_cfg, "checkpoint_bundle_v2_subdir", "") or "").strip()
                        if not v2_subdir:
                            v2_subdir = str(getattr(self.sim_cfg, "checkpoint_subdir", "checkpoint_bundles_v2") or "checkpoint_bundles_v2")
                        v2_strict = bool(getattr(self.sim_cfg, "checkpoint_bundle_v2_strict_replay_mode", True))

                        root = os.path.join(self.run_dir, v2_subdir)
                        os.makedirs(root, exist_ok=True)

                        setattr(self, "_current_gen", int(gen))
                        next_gen = int(gen) + 1
                        bpath = os.path.join(root, f"bundle_gen{next_gen:06d}")

                        if not bool(bundle_v2_saved_this_boundary):
                            t_checkpoint_interrupt0 = _perf_time(self)
                            checkpoint_bundle_v2_save(self, bpath, generation_index=int(next_gen), strict_replay_mode=v2_strict)
                            _perf_add_checkpoint_details(self, buckets)
                            rotate_checkpoint_bundles_v2(root, keep_last=rotate)
                            checkpoint_elapsed = _perf_elapsed(self, t_checkpoint_interrupt0)
                            buckets["checkpoint_total"] = float(buckets.get("checkpoint_total", 0.0)) + checkpoint_elapsed
                            buckets["checkpoint_interrupt_total"] = float(
                                buckets.get("checkpoint_interrupt_total", 0.0)
                            ) + checkpoint_elapsed
                            bundle_v2_saved_this_boundary = True
                            bundle_v2_saved_path = str(bpath)

                        # Always print a deterministic path for operators.
                        _pp = str(bundle_v2_saved_path or bpath)
                        print(f"[interrupt] checkpoint bundle v2 saved: {_pp}", flush=True)
                    except Exception as _int_ckpt_exc:
                        _raise_native_checkpoint_failure(
                            self,
                            _int_ckpt_exc,
                            stage="checkpoint_graceful_interrupt",
                        )
                        try:
                            print(f"[interrupt] WARNING: failed to save interrupt checkpoint bundle v2 at gen {gen}: {_int_ckpt_exc}", flush=True)
                        except Exception:
                            pass
                    break

                if self.perf:
                    # Write perf buckets in a stable long format for later analysis.
                    try:
                        if hasattr(self, "_perf_writer") and self._perf_writer is not None:
                            self._perf_writer.write_generation(gen=gen, total_s=time.perf_counter() - gen_t0, buckets=buckets)
                    except Exception:
                        pass

                if self.perf:
                    total = time.perf_counter() - gen_t0
                    print(f"[perf] gen {gen} timing (steps={steps_ran}) total={total:.3f}s")
                    for k, v in sorted(buckets.items(), key=lambda kv: kv[1], reverse=True):
                        pct = (100.0 * v / total) if total > 0 else 0.0
                        print(f"  {k:16s} {v:6.3f}s  ({pct:5.1f}%)")
            # -----------------------------
            # Phase 5d: finalize run status (observability only)
            # -----------------------------
            try:
                if _viz is not None and getattr(_viz, "enabled", False):
                    _viz.write_run_status(
                        status="complete",
                        current_generation=int(_viz_last_completed_gen) + 1,
                        target_generation=int(end_gen),
                        last_error=None,
                        branch_state="COMPLETE",
                        run_kind=str(_viz_run_kind),
                    )
            except Exception:
                pass
            _slot_terminal_status = "complete"
        except Exception as _run_exc:
            # Best-effort: mark run failed for UI observability, then re-raise.
            try:
                if _viz is not None and getattr(_viz, "enabled", False):
                    _viz.write_run_status(
                        status="failed",
                        current_generation=int(_viz_last_completed_gen) + 1,
                        target_generation=int(end_gen),
                        last_error=str(_run_exc),
                        branch_state="FAILED",
                        run_kind=str(_viz_run_kind),
                    )
            except Exception:
                pass
            _slot_terminal_error = f"{type(_run_exc).__name__}: {_run_exc}"
            raise
        finally:
            if _slot_lease is not None:
                try:
                    _slot_lease.close(
                        status=_slot_terminal_status,
                        error=_slot_terminal_error,
                    )
                except Exception:
                    pass
            try:
                chronicle = getattr(self, "scientific_chronicle", None)
                if chronicle is not None:
                    chronicle.finalize(status="complete")
            except Exception:
                pass
            try:
                bo = getattr(self, "branch_observability", None)
                if bo is not None:
                    bo.finalize(
                        self,
                        generation=int(getattr(self, "_current_gen", int(end_gen)) or int(end_gen)),
                        tick=int(getattr(self, "_tick", getattr(self, "_pred_tick", 0)) or 0),
                        status="complete",
                    )
            except Exception:
                pass

            # Every successfully completed Python run receives the same compact
            # terminal Canopy bundle as the native owner.  It is a read-only
            # identity snapshot, independent of opt-in event observability.
            try:
                run_dir = str(getattr(self, "run_dir", "") or "").strip()
                if _slot_terminal_status in {"complete", "failed"} and run_dir:
                    from tools.visualization.living_lineage_canopy.python.default_run_export import (  # noqa: WPS433
                        materialize_default_terminal_canopy_from_agents,
                    )

                    run_id = str(
                        getattr(getattr(self, "sim_cfg", None), "campaign_id", "")
                        or Path(run_dir).name
                    )
                    self._default_canopy_materialization = (
                        materialize_default_terminal_canopy_from_agents(
                            run_root=Path(run_dir),
                            run_id=run_id,
                            terminal_generation=int(
                                getattr(self, "_current_gen", int(end_gen) - 1)
                            )
                            + 1,
                            agents=self.pop.alive_agents(),
                            source_state=("source_complete" if _slot_terminal_status == "complete" else "source_interrupted"),
                        )
                    )
            except Exception as exc:
                self._default_canopy_materialization_error = (
                    f"{type(exc).__name__}: {exc}"
                )

            # Optional compact hotpath diagnostic summary. Metrics-only and
            # default-off; failures must never affect simulation closeout.
            try:
                _envhot_diag = getattr(self, "env_hotpath_expression_diagnostics", None)
                if _envhot_diag is not None:
                    _envhot_diag.write_summary()
                    if bool(getattr(_envhot_diag, "decomposition_enabled", False)):
                        _envhot_diag.write_decomposition_summary()
                    if bool(getattr(_envhot_diag, "phenotype_decomposition_enabled", False)):
                        _envhot_diag.write_phenotype_decomposition_summary()
                    if bool(getattr(_envhot_diag, "developmental_residual_enabled", False)):
                        _envhot_diag.write_developmental_residual_summary()
                    if bool(getattr(_envhot_diag, "developmental_cache_equality_enabled", False)):
                        _envhot_diag.write_developmental_cache_equality_summary()
                    if bool(getattr(_envhot_diag, "developmental_effective_projection_enabled", False)):
                        _envhot_diag.write_developmental_effective_projection_summary()
            except Exception:
                pass

            # Close Tier 2 drought log writers (best-effort).
            try:
                self.env.spatial_droughts_close()
            except Exception:
                pass

            # Clear membership tracing context (best-effort).
            try:
                from simulation.membership_trace import clear_trace_context  # noqa: WPS433
                clear_trace_context()
            except Exception:
                pass

            # Restore original signal handlers to avoid leaking into callers/tests.
            try:
                signal.signal(signal.SIGINT, old_sigint)
                signal.signal(signal.SIGTERM, old_sigterm)
            except Exception:
                pass
