from __future__ import annotations

import hashlib
import json
import time
from typing import Any


DEVELOPMENTAL_PROGRAM_SCHEMA_VERSION = "developmental_program_v1"
DEFAULT_STAGE_ID = "default"


def _as_nonneg_int(value: Any, default: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = int(default)
    if v < 0:
        v = int(default)
    return int(v)


def _as_stage_id(value: Any) -> str:
    s = str(value or "").strip()
    return s or DEFAULT_STAGE_ID


def _as_end_step(value: Any) -> int:
    # -1 means open-ended.
    try:
        v = int(value)
    except Exception:
        v = -1
    if v < -1:
        v = -1
    return int(v)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        s = str(value).strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(default)


def _strict_fail(code: str, context: str) -> None:
    raise RuntimeError(
        "[DETERMINISM_VIOLATION]\n"
        f"code={str(code)}\n"
        f"context={str(context)}\n"
    )


def _developmental_residual_diag_payload(raw: Any) -> tuple[Any, str, float]:
    try:
        from simulation.env_hotpath_expression_diagnostics import current_diagnostics, stable_payload_key  # noqa: WPS433

        diag = current_diagnostics()
        if diag is None or not bool(getattr(diag, "developmental_residual_enabled", False)):
            return None, "", 0.0
        payload_key, _payload_bytes = stable_payload_key(raw)
        return diag, str(payload_key), time.perf_counter()
    except Exception:
        return None, "", 0.0


def default_program() -> dict[str, Any]:
    return {
        "schema_version": DEVELOPMENTAL_PROGRAM_SCHEMA_VERSION,
        "stages": [{"stage_id": DEFAULT_STAGE_ID, "start_step": 0, "end_step": -1}],
        "module_windows": [],
        "lock_commitments": [],
        "lock_dependencies": [],
    }


def canonicalize_developmental_program(raw: Any) -> dict[str, Any]:
    _diag = None
    _payload_key = ""
    _payload_bytes = 0
    _t0 = 0.0
    try:
        from simulation.env_hotpath_expression_diagnostics import current_diagnostics, stable_payload_key  # noqa: WPS433

        _diag = current_diagnostics()
        if _diag is not None:
            _payload_key, _payload_bytes = stable_payload_key(raw)
            _t0 = time.perf_counter()
    except Exception:
        _diag = None

    def _record(out: dict[str, Any]) -> dict[str, Any]:
        if _diag is not None:
            try:
                _diag.record_canonicalize_call(
                    payload_key=str(_payload_key),
                    payload_bytes=int(_payload_bytes),
                    seconds=time.perf_counter() - _t0,
                )
            except Exception:
                pass
        return out

    if not isinstance(raw, dict):
        return _record(default_program())

    stages_raw = raw.get("stages", [])
    stages: list[dict[str, Any]] = []
    if isinstance(stages_raw, list):
        for rec in stages_raw:
            if not isinstance(rec, dict):
                continue
            stage_id = _as_stage_id(rec.get("stage_id", DEFAULT_STAGE_ID))
            start_step = _as_nonneg_int(rec.get("start_step", 0), 0)
            end_step = _as_end_step(rec.get("end_step", -1))
            if end_step >= 0 and end_step < start_step:
                end_step = int(start_step)
            stages.append({"stage_id": stage_id, "start_step": int(start_step), "end_step": int(end_step)})
    if not stages:
        stages = [{"stage_id": DEFAULT_STAGE_ID, "start_step": 0, "end_step": -1}]
    stages.sort(key=lambda r: (int(r["start_step"]), int(r["end_step"]), str(r["stage_id"])))

    mw_raw = raw.get("module_windows", [])
    module_windows: list[dict[str, Any]] = []
    if isinstance(mw_raw, list):
        for rec in mw_raw:
            if not isinstance(rec, dict):
                continue
            module_id = str(rec.get("module_id", "") or "").strip()
            if not module_id:
                continue
            stage_id = _as_stage_id(rec.get("stage_id", DEFAULT_STAGE_ID))
            module_windows.append(
                {
                    "module_id": module_id,
                    "stage_id": stage_id,
                    "active": bool(rec.get("active", True)),
                }
            )
    module_windows.sort(key=lambda r: (str(r["module_id"]), str(r["stage_id"])))

    lc_raw = raw.get("lock_commitments", [])
    lock_commitments: list[dict[str, Any]] = []
    if isinstance(lc_raw, list):
        for rec in lc_raw:
            if not isinstance(rec, dict):
                continue
            module_id = str(rec.get("module_id", "") or "").strip()
            if not module_id:
                continue
            stage_id = _as_stage_id(rec.get("stage_id", DEFAULT_STAGE_ID))
            lock_commitments.append(
                {
                    "module_id": module_id,
                    "stage_id": stage_id,
                    "locked_active": _as_bool(rec.get("locked_active", rec.get("active", True)), True),
                }
            )
    lock_commitments.sort(key=lambda r: (str(r["stage_id"]), str(r["module_id"]), int(bool(r["locked_active"]))))
    lock_commitments_dedup: list[dict[str, Any]] = []
    seen_lock_keys: set[tuple[str, str]] = set()
    for rec in lock_commitments:
        key = (str(rec["stage_id"]), str(rec["module_id"]))
        if key in seen_lock_keys:
            continue
        seen_lock_keys.add(key)
        lock_commitments_dedup.append(rec)

    ld_raw = raw.get("lock_dependencies", [])
    lock_dependencies: list[dict[str, Any]] = []
    if isinstance(ld_raw, list):
        for rec in ld_raw:
            if not isinstance(rec, dict):
                continue
            module_id = str(rec.get("module_id", "") or "").strip()
            depends_on = str(rec.get("depends_on_module_id", "") or "").strip()
            if not module_id or not depends_on:
                continue
            lock_dependencies.append(
                {
                    "module_id": module_id,
                    "depends_on_module_id": depends_on,
                    "requires_active": _as_bool(rec.get("requires_active", True), True),
                }
            )
    lock_dependencies.sort(
        key=lambda r: (
            str(r["module_id"]),
            str(r["depends_on_module_id"]),
            int(bool(r["requires_active"])),
        )
    )
    lock_dependencies_dedup: list[dict[str, Any]] = []
    seen_dep_keys: set[tuple[str, str]] = set()
    for rec in lock_dependencies:
        key = (str(rec["module_id"]), str(rec["depends_on_module_id"]))
        if key in seen_dep_keys:
            continue
        seen_dep_keys.add(key)
        lock_dependencies_dedup.append(rec)

    return _record({
        "schema_version": DEVELOPMENTAL_PROGRAM_SCHEMA_VERSION,
        "stages": stages,
        "module_windows": module_windows,
        "lock_commitments": lock_commitments_dedup,
        "lock_dependencies": lock_dependencies_dedup,
    })


def resolve_stage_id(program: dict[str, Any] | None, *, age_steps: int) -> str:
    _diag, _payload_key, _t0 = _developmental_residual_diag_payload(program)
    p = canonicalize_developmental_program(program)
    age = max(0, int(age_steps))
    out = DEFAULT_STAGE_ID
    for rec in p.get("stages", []):
        start = int(rec.get("start_step", 0) or 0)
        end = int(rec.get("end_step", -1) or -1)
        if age < start:
            continue
        if end >= 0 and age > end:
            continue
        out = str(rec.get("stage_id", DEFAULT_STAGE_ID) or DEFAULT_STAGE_ID)
        break
    if _diag is not None:
        try:
            _diag.record_developmental_stage_resolution(
                program_key=str(_payload_key),
                age_steps=int(age),
                stage_id=str(out),
                seconds=time.perf_counter() - _t0,
            )
        except Exception:
            pass
    return str(out)


def resolve_module_active(
    program: dict[str, Any] | None,
    *,
    module_id: str,
    age_steps: int,
) -> tuple[bool, str, str]:
    _diag, _payload_key, _t0 = _developmental_residual_diag_payload(program)
    p = canonicalize_developmental_program(program)
    stage_id = resolve_stage_id(p, age_steps=int(age_steps))
    rules = [
        r for r in (p.get("module_windows", []) or [])
        if str(r.get("module_id", "")) == str(module_id)
    ]
    if not rules:
        out = (True, "default_active_no_window", stage_id)
    else:
        out = (False, "inactive_by_stage", stage_id)
        for rec in rules:
            if str(rec.get("stage_id", DEFAULT_STAGE_ID)) == stage_id:
                active = bool(rec.get("active", True))
                out = (active, ("active_by_stage" if active else "inactive_by_stage"), stage_id)
                break
    if _diag is not None:
        try:
            _diag.record_developmental_module_active(
                program_key=str(_payload_key),
                age_steps=max(0, int(age_steps)),
                module_id=str(module_id or ""),
                stage_id=str(stage_id),
                seconds=time.perf_counter() - _t0,
            )
        except Exception:
            pass
    return out


def module_structure_signature(modules: list[dict[str, Any]] | None) -> str:
    out = []
    for m in sorted(list(modules or []), key=lambda x: str(x.get("module_id", ""))):
        out.append(
            {
                "module_id": str(m.get("module_id", "")),
                "module_type": str(m.get("module_type", "")),
            }
        )
    raw = json.dumps(out, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def lock_commitments_for_stage(program: dict[str, Any] | None, *, stage_id: str) -> list[dict[str, Any]]:
    p = canonicalize_developmental_program(program)
    sid = _as_stage_id(stage_id)
    out = [
        {
            "module_id": str(r.get("module_id", "")),
            "stage_id": sid,
            "locked_active": bool(r.get("locked_active", True)),
        }
        for r in (p.get("lock_commitments", []) or [])
        if str(r.get("stage_id", DEFAULT_STAGE_ID)) == sid and str(r.get("module_id", "")).strip()
    ]
    out.sort(key=lambda r: (str(r["module_id"]), int(bool(r["locked_active"]))))
    return out


def lock_dependencies_for_module(program: dict[str, Any] | None, *, module_id: str) -> list[dict[str, Any]]:
    p = canonicalize_developmental_program(program)
    mid = str(module_id or "").strip()
    out = [
        {
            "module_id": mid,
            "depends_on_module_id": str(r.get("depends_on_module_id", "")),
            "requires_active": bool(r.get("requires_active", True)),
        }
        for r in (p.get("lock_dependencies", []) or [])
        if str(r.get("module_id", "")) == mid and str(r.get("depends_on_module_id", "")).strip()
    ]
    out.sort(key=lambda r: (str(r["depends_on_module_id"]), int(bool(r["requires_active"]))))
    return out


def apply_irreversible_stage_locks(
    *,
    program: dict[str, Any] | None,
    stage_id: str,
    prev_stage_id: str,
    active_by_module_id: dict[str, Any] | None,
    inactive_reason_by_module_id: dict[str, Any] | None,
    previous_lock_state: dict[str, Any] | None,
    strict: bool = False,
) -> dict[str, Any]:
    _diag, _payload_key, _t0 = _developmental_residual_diag_payload(program)
    p = canonicalize_developmental_program(program)
    stage = _as_stage_id(stage_id)
    prev_stage = _as_stage_id(prev_stage_id)

    active: dict[str, bool] = {
        str(k): bool(v)
        for k, v in (active_by_module_id or {}).items()
        if str(k).strip()
    }
    reasons: dict[str, str] = {
        str(k): str(v or "")
        for k, v in (inactive_reason_by_module_id or {}).items()
        if str(k).strip()
    }
    locks: dict[str, bool] = {
        str(k): bool(v)
        for k, v in (previous_lock_state or {}).items()
        if str(k).strip()
    }
    locks = dict(sorted(locks.items(), key=lambda kv: str(kv[0])))
    events: list[dict[str, Any]] = []
    entered_stage = bool(stage != prev_stage)

    if entered_stage:
        for rec in lock_commitments_for_stage(p, stage_id=stage):
            mid = str(rec.get("module_id", "") or "")
            target = bool(rec.get("locked_active", True))
            if not mid:
                continue
            if mid in locks:
                current = bool(locks.get(mid, False))
                if current != target:
                    if strict:
                        _strict_fail(
                            "DET_DEV_D2_ILLEGAL_REVERSE_LOCK",
                            f"stage_id={stage};module_id={mid};locked={int(current)};requested={int(target)}",
                        )
                    events.append(
                        {
                            "event": "illegal_reverse_lock_ignored",
                            "module_id": mid,
                            "existing_locked_active": bool(current),
                            "requested_locked_active": bool(target),
                        }
                    )
                continue

            dependency_conflict = None
            if target:
                for dep in lock_dependencies_for_module(p, module_id=mid):
                    dep_mid = str(dep.get("depends_on_module_id", "") or "")
                    req = bool(dep.get("requires_active", True))
                    dep_state = bool(locks.get(dep_mid, active.get(dep_mid, False)))
                    if dep_state != req:
                        dependency_conflict = (dep_mid, req, dep_state)
                        break
            if dependency_conflict is not None:
                dep_mid, req, dep_state = dependency_conflict
                locks[mid] = False
                events.append(
                    {
                        "event": "dependency_conflict_forced_inactive",
                        "module_id": mid,
                        "depends_on_module_id": dep_mid,
                        "requires_active": bool(req),
                        "observed_active": bool(dep_state),
                    }
                )
            else:
                locks[mid] = bool(target)
                events.append(
                    {
                        "event": "lock_committed",
                        "module_id": mid,
                        "locked_active": bool(target),
                    }
                )

    for mid in sorted(active.keys(), key=lambda x: str(x)):
        if mid not in locks:
            continue
        locked_active = bool(locks[mid])
        active[mid] = bool(locked_active)
        reasons[mid] = "active_by_irreversible_lock" if locked_active else "inactive_by_irreversible_lock"

    out = {
        "stage_id": stage,
        "entered_stage": bool(entered_stage),
        "active_by_module_id": dict(sorted(active.items(), key=lambda kv: str(kv[0]))),
        "inactive_reason_by_module_id": dict(sorted(reasons.items(), key=lambda kv: str(kv[0]))),
        "lock_state_by_module_id": dict(sorted(locks.items(), key=lambda kv: str(kv[0]))),
        "events": list(events),
    }
    if _diag is not None:
        try:
            _diag.record_developmental_lock_application(
                program_key=str(_payload_key),
                stage_id=str(stage),
                prev_stage_id=str(prev_stage),
                module_count=int(len(active)),
                seconds=time.perf_counter() - _t0,
            )
        except Exception:
            pass
    return out
