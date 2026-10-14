from __future__ import annotations

from typing import Any, Dict, Optional, List
import time
import hashlib
import os
import torch

from engine.simulation.forensic_v4_probes import emit_policy_bank_digest, tensor_sha256_cpu_float32, emit_policy_bank_sync_digest


class BankMixin:
    """Parameter bank management mixin.

    Performance goals
    -----------------
    - Avoid per-tick bank rebuilds: banks only change on birth/mutation
      events (slot_token changes).
    - Make per-tick checks cheap by using a coarse dirty flag.

    This keeps correctness identical: if banks are dirty, we sync; if not,
    the banks are already consistent with current agent slot_tokens.
    """

    def _mark_bank_dirty(self) -> None:
        """Mark MLP and GRU parameter banks as needing sync."""
        setattr(self, "_bank_dirty_mlp", True)
        setattr(self, "_bank_dirty_gru", True)

        # FDP v3: track dirty-mark callsites to catch missing dirty marks on resume.
        try:
            v = str(os.getenv("EIEL_FORENSIC_V4", "0")).strip()
            if v in ("1", "true", "TRUE", "yes", "YES"):
                import inspect
                frm = None
                try:
                    frm = inspect.stack()[1]
                except Exception:
                    frm = None
                rec = {
                    "gen": int(getattr(self, "_eiel_fdp_cur_gen", -1)),
                    "tick": int(getattr(self, "_eiel_fdp_cur_tick", -1)),
                }
                if frm is not None:
                    try:
                        rec["fn"] = str(getattr(frm, "function", ""))
                        rec["file"] = str(getattr(frm, "filename", "")).split("/")[-1]
                        rec["lineno"] = int(getattr(frm, "lineno", -1))
                    except Exception:
                        pass
                lst = getattr(self, "_eiel_dirty_mark_events", None)
                if not isinstance(lst, list):
                    lst = []
                lst.append(rec)
                if len(lst) > 64:
                    lst = lst[-64:]
                setattr(self, "_eiel_dirty_mark_events", lst)
                setattr(self, "_eiel_dirty_mark_count", int(getattr(self, "_eiel_dirty_mark_count", 0) or 0) + 1)
        except Exception:
            pass



    # -------------------------
    # Forensic V4 probes (A)
    # -------------------------
    def _eiel_collect_bank_info(self) -> Dict[str, Any]:
        """Collect stable digests for policy parameter banks.

        This MUST NOT draw RNG and MUST NOT mutate state.
        """
        info: Dict[str, Any] = {}
        try:
            info['bank_device'] = str(getattr(self, '_bank_device', ''))
            info['bank_dtype'] = str(getattr(self, '_bank_dtype', ''))
            info['bank_dirty_mlp'] = bool(getattr(self, '_bank_dirty_mlp', False))
            info['bank_dirty_gru'] = bool(getattr(self, '_bank_dirty_gru', False))
        except Exception:
            pass

        # Bank shapes + row-level and active-slot digests (high-signal forensic probes).
        # These MUST NOT draw RNG and MUST NOT mutate state.
        def _row_sha_list(t: Any, n: int = 8) -> List[str]:
            out: List[str] = []
            try:
                if t is None:
                    return out
                tt = t.detach().cpu()
                # Work in float32 for stable hashing across device/dtype.
                if hasattr(tt, "to"):
                    tt = tt.to(dtype=torch.float32)
                rows = int(getattr(tt, "shape", [0])[0]) if getattr(tt, "shape", None) is not None else 0
                for i in range(min(int(n), max(0, rows))):
                    out.append(tensor_sha256_cpu_float32(tt[i : i + 1]))
            except Exception:
                return out
            return out

        def _active_slots_sorted() -> List[int]:
            slots: List[int] = []
            try:
                agents = getattr(getattr(self, "pop", None), "agents", None)
                if isinstance(agents, dict):
                    for a in agents.values():
                        if not getattr(a, "alive", True):
                            continue
                        s = int(getattr(a, "slot", -1))
                        if s >= 0:
                            slots.append(s)
            except Exception:
                pass
            slots.sort()
            return slots

        def _active_slots_by_policy():
            """Return alive slots split by policy type (mlp vs gru)."""
            mlp: List[int] = []
            gru: List[int] = []
            try:
                agents = getattr(getattr(self, "pop", None), "agents", None)
                if isinstance(agents, dict):
                    for a in agents.values():
                        if not getattr(a, "alive", True):
                            continue
                        s = int(getattr(a, "slot", -1))
                        if s < 0:
                            continue
                        pol = getattr(a, "policy", None)
                        if pol is not None and hasattr(pol, "net"):
                            mlp.append(s)
                        if pol is not None and hasattr(pol, "gru"):
                            gru.append(s)
            except Exception:
                pass
            mlp.sort()
            gru.sort()
            return mlp, gru


        def _hash_active_rows(t: Any, slots: List[int]) -> str:
            try:
                if t is None or not slots:
                    return ""
                tt = t.detach().cpu().to(dtype=torch.float32)
                idx = torch.tensor(slots, dtype=torch.long)
                sel = tt.index_select(0, idx)
                return tensor_sha256_cpu_float32(sel)
            except Exception:
                return "error"

        def _hash_active_rows_multiset(t: Any, slots: List[int]) -> str:
            """Order-insensitive hash of active rows.
            Useful to detect pure row-permutation differences vs true value differences.
            """
            try:
                if t is None or not slots:
                    return ""
                tt = t.detach().cpu().to(dtype=torch.float32)
                digests: List[str] = []
                for s in slots:
                    digests.append(tensor_sha256_cpu_float32(tt[s : s + 1]))
                digests.sort()
                b = ("|".join(digests)).encode("utf-8")
                return __import__("hashlib").sha256(b).hexdigest()
            except Exception:
                return "error"

        def _hash_active_slot_rowpairs(t: Any, slots: List[int]) -> str:
            """Slot->row digest mapping hash (order-insensitive via sorted slots)."""
            try:
                if t is None or not slots:
                    return ""
                tt = t.detach().cpu().to(dtype=torch.float32)
                pairs: List[str] = []
                for s in slots:
                    pairs.append(f"{int(s)}:{tensor_sha256_cpu_float32(tt[s : s + 1])}")
                # slots is already sorted, but keep it explicit and stable
                pairs.sort()
                b = ("|".join(pairs)).encode("utf-8")
                return __import__("hashlib").sha256(b).hexdigest()
            except Exception:
                return "error"
        # Hash core packed banks (fastest high-signal digest).
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        try:
            info['mlp_pack_sha256'] = tensor_sha256_cpu_float32(getattr(self, '_mlp_pack', None))
        except Exception:
            info['mlp_pack_sha256'] = 'error'

        # Extra MLP bank forensic fields
        try:
            mp = getattr(self, '_mlp_pack', None)
            info['mlp_pack_shape'] = list(getattr(mp, 'shape', [])) if mp is not None else []
            info['mlp_pack_first8_rows_sha256'] = _row_sha_list(mp, 8)
            act = _active_slots_sorted()
            info['active_slots_count'] = int(len(act))
            # Hash the ordered active slot list itself (helps detect ordering/capacity issues)
            try:
                b = (','.join(str(x) for x in act)).encode('utf-8')
                info['active_slots_sha256'] = __import__('hashlib').sha256(b).hexdigest()
            except Exception:
                info['active_slots_sha256'] = 'error'
            info['mlp_pack_active_rows_sha256'] = _hash_active_rows(mp, act)
            # FDP: also hash alive agents' slot_token multiset to detect collisions/drift.
            try:
                _alive = list(getattr(self.pop, 'alive_agents')())
            except Exception:
                _alive = []
            try:
                _st = []
                _sl = []
                for _a in _alive:
                    try:
                        _st.append(int(getattr(_a, 'slot_token', -1)))
                    except Exception:
                        _st.append(-1)
                    try:
                        _sl.append(int(getattr(_a, 'slot', -1)))
                    except Exception:
                        _sl.append(-1)
                _st_sorted = sorted(_st)
                _sl_sorted = sorted(_sl)
                info['alive_slot_tokens_sha256'] = __import__('hashlib').sha256((','.join(str(x) for x in _st_sorted)).encode('utf-8')).hexdigest()
                info['alive_slots_sha256'] = __import__('hashlib').sha256((','.join(str(x) for x in _sl_sorted)).encode('utf-8')).hexdigest()
                # duplicate count (slot_token collisions are catastrophic)
                _dupes = 0
                for _i in range(1, len(_st_sorted)):
                    if _st_sorted[_i] == _st_sorted[_i-1] and _st_sorted[_i] >= 0:
                        _dupes += 1
                info['alive_slot_token_dupe_count'] = int(_dupes)
            except Exception:
                pass
            info['mlp_pack_active_rows_multiset_sha256'] = _hash_active_rows_multiset(mp, act)
            info['mlp_pack_active_slot_rowpairs_sha256'] = _hash_active_slot_rowpairs(mp, act)
            # FDP v3: policy-kind filtered active-slot hashes (reduce noise from unused cache rows).
            try:
                mlp_act, gru_act = _active_slots_by_policy()
                info['mlp_slots_count'] = int(len(mlp_act))
                info['gru_slots_count'] = int(len(gru_act))
                try:
                    b = (','.join(str(x) for x in mlp_act)).encode('utf-8')
                    info['mlp_slots_sha256'] = hashlib.sha256(b).hexdigest()
                except Exception:
                    info['mlp_slots_sha256'] = 'error'
                try:
                    b = (','.join(str(x) for x in gru_act)).encode('utf-8')
                    info['gru_slots_sha256'] = hashlib.sha256(b).hexdigest()
                except Exception:
                    info['gru_slots_sha256'] = 'error'
                info['mlp_pack_active_rows_multiset_sha256_mlp_only'] = _hash_active_rows_multiset(mp, mlp_act)
                info['mlp_pack_active_slot_rowpairs_sha256_mlp_only'] = _hash_active_slot_rowpairs(mp, mlp_act)
            except Exception:
                pass
        except Exception:
            pass

        # GRU pack + memory (if initialized).
        try:
            info['gru_inited'] = bool(getattr(self, '_gru_inited', False))
        except Exception:
            info['gru_inited'] = False
        if info.get('gru_inited', False):
            act = _active_slots_sorted()
            try:
                info['gru_Hmem_sha256'] = tensor_sha256_cpu_float32(getattr(self, '_Hmem', None))
            except Exception:
                info['gru_Hmem_sha256'] = 'error'

            # Extra GRU bank forensic fields
            try:
                hm = getattr(self, '_Hmem', None)
                info['gru_Hmem_shape'] = list(getattr(hm, 'shape', [])) if hm is not None else []
                info['gru_Hmem_first8_rows_sha256'] = _row_sha_list(hm, 8)
                act = _active_slots_sorted()
                info['gru_Hmem_active_rows_sha256'] = _hash_active_rows(hm, act)
                info['gru_Hmem_active_rows_multiset_sha256'] = _hash_active_rows_multiset(hm, act)
                info['gru_Hmem_active_slot_rowpairs_sha256'] = _hash_active_slot_rowpairs(hm, act)
            except Exception:
                pass

            try:
                info['gru_Wih_sha256'] = tensor_sha256_cpu_float32(getattr(self, '_Wih', None))
                info['gru_Whh_sha256'] = tensor_sha256_cpu_float32(getattr(self, '_Whh', None))
                info['gru_Wih_active_rows_multiset_sha256'] = _hash_active_rows_multiset(getattr(self, '_Wih', None), act)
                info['gru_Wih_active_slot_rowpairs_sha256'] = _hash_active_slot_rowpairs(getattr(self, '_Wih', None), act)
                info['gru_Whh_active_rows_multiset_sha256'] = _hash_active_rows_multiset(getattr(self, '_Whh', None), act)
                info['gru_Whh_active_slot_rowpairs_sha256'] = _hash_active_slot_rowpairs(getattr(self, '_Whh', None), act)
                # FDP v3: policy-kind filtered active-slot hashes for GRU bank (ignore unused rows).
                try:
                    _mlp_act, _gru_act = _active_slots_by_policy()
                    info['gru_Wih_active_rows_multiset_sha256_gru_only'] = _hash_active_rows_multiset(getattr(self, '_Wih', None), _gru_act)
                    info['gru_Wih_active_slot_rowpairs_sha256_gru_only'] = _hash_active_slot_rowpairs(getattr(self, '_Wih', None), _gru_act)
                    info['gru_Whh_active_rows_multiset_sha256_gru_only'] = _hash_active_rows_multiset(getattr(self, '_Whh', None), _gru_act)
                    info['gru_Whh_active_slot_rowpairs_sha256_gru_only'] = _hash_active_slot_rowpairs(getattr(self, '_Whh', None), _gru_act)
                    info['gru_Hmem_active_rows_multiset_sha256_gru_only'] = _hash_active_rows_multiset(getattr(self, '_Hmem', None), _gru_act)
                    info['gru_Hmem_active_slot_rowpairs_sha256_gru_only'] = _hash_active_slot_rowpairs(getattr(self, '_Hmem', None), _gru_act)
                    info['gru_pack_active_rows_multiset_sha256_gru_only'] = _hash_active_rows_multiset(getattr(self, '_gru_pack', None), _gru_act)
                    info['gru_pack_active_slot_rowpairs_sha256_gru_only'] = _hash_active_slot_rowpairs(getattr(self, '_gru_pack', None), _gru_act)
                except Exception:
                    pass
            except Exception:
                pass
        return info

    def _eiel_emit_policy_bank_probe(self, *, tag: str, generation_index: Optional[int], tick: Optional[int]) -> None:
        """Emit a policy/bank digest record to the run debug folder."""
        try:
            run_dir = str(getattr(self, 'run_dir', ''))
            if not run_dir:
                return
            info = self._eiel_collect_bank_info()

            # FDP v3: around gen3 tick100-110 bank_pre_forward, record alive slot deltas and added-slot bank/expected digests.
            try:
                if str(tag).endswith('.bank_pre_forward') and generation_index is not None and tick is not None:
                    if int(generation_index) == 3 and (100 <= int(tick) <= 110):
                        try:
                            _alive = list(getattr(self.pop, 'alive_agents')())
                        except Exception:
                            _alive = []
                        cur_slots = []
                        for _a in _alive:
                            try:
                                s = int(getattr(_a, 'slot', -1))
                                if s >= 0:
                                    cur_slots.append(s)
                            except Exception:
                                continue
                        cur_set = set(cur_slots)
                        prev_map = getattr(self, '_eiel_prev_alive_slots_for_bank_probe', None)
                        if not isinstance(prev_map, dict):
                            prev_map = {}
                        prev_set = set(prev_map.get(int(generation_index), set()) or set())
                        added = sorted(cur_set - prev_set)[:32]
                        removed = sorted(prev_set - cur_set)[:32]
                        info['alive_slots_prev_count'] = int(len(prev_set))
                        info['alive_slots_cur_count'] = int(len(cur_set))
                        info['alive_slots_added'] = added
                        info['alive_slots_removed'] = removed

                        # Per-added-slot details (sample)
                        added_details = []

                        # last-seen map (for reuse diagnostics)
                        last_seen = getattr(self, '_eiel_slot_last_seen', None)
                        if not isinstance(last_seen, dict):
                            last_seen = {}

                        def _mlp_row_sha(slot: int) -> str:
                            try:
                                if getattr(self, '_mlp_pack', None) is None:
                                    return ''
                                ss = int(slot)
                                if ss < 0:
                                    return ''
                                if ss >= int(getattr(self._mlp_pack, 'shape', [0])[0]):
                                    return ''
                                row = self._mlp_pack[ss].detach().to('cpu').contiguous().float()
                                return hashlib.sha256(row.numpy().tobytes()).hexdigest()
                            except Exception:
                                return ''

                        for s in added[:8]:
                            try:
                                # Find the alive agent owning this slot
                                owner = None
                                for _a in _alive:
                                    try:
                                        if int(getattr(_a, 'slot', -1)) == int(s):
                                            owner = _a
                                            break
                                    except Exception:
                                        continue
                                if owner is None:
                                    continue
                                aid = int(getattr(owner, 'agent_id', -1))
                                st = int(getattr(owner, 'slot_token', 0))
                                bt = int(getattr(owner, '_bank_token', -1))
                                gbt = int(getattr(owner, '_gru_bank_token', -1))
                                bank_sha = _mlp_row_sha(int(s))
                                exp_sha = ''
                                try:
                                    net = getattr(getattr(owner, 'policy', None), 'net', None)
                                    if net is not None:
                                        w1 = net[0].weight.detach().cpu().reshape(-1)
                                        b1 = net[0].bias.detach().cpu().reshape(-1)
                                        w2 = net[2].weight.detach().cpu().reshape(-1)
                                        b2 = net[2].bias.detach().cpu().reshape(-1)
                                        w3 = net[4].weight.detach().cpu().reshape(-1)
                                        b3 = net[4].bias.detach().cpu().reshape(-1)
                                        pack = torch.cat([w1, b1, w2, b2, w3, b3], dim=0).contiguous().float()
                                        exp_sha = hashlib.sha256(pack.numpy().tobytes()).hexdigest()
                                except Exception:
                                    exp_sha = ''

                                # Hmem row sha for GRU reset diagnostics
                                hmem_sha = ''
                                try:
                                    if getattr(self, '_Hmem', None) is not None and int(s) >= 0 and int(s) < int(getattr(self._Hmem, 'shape', [0])[0]):
                                        row = self._Hmem[int(s)].detach().to('cpu').contiguous().float()
                                        hmem_sha = hashlib.sha256(row.numpy().tobytes()).hexdigest()
                                except Exception:
                                    hmem_sha = ''

                                # GRU weight/pack row sha (for slots owned by GRU policies)
                                policy_kind = 'unknown'
                                gru_bank_pack_sha = ''
                                gru_expected_pack_sha = ''
                                gru_bank_wih_sha = ''
                                gru_expected_wih_sha = ''
                                gru_bank_whh_sha = ''
                                gru_expected_whh_sha = ''
                                try:
                                    pol = getattr(owner, 'policy', None)
                                    if pol is not None and getattr(pol, 'gru', None) is not None:
                                        policy_kind = 'gru'
                                        try:
                                            if getattr(self, '_gru_pack', None) is not None and int(s) >= 0 and int(s) < int(getattr(self._gru_pack, 'shape', [0])[0]):
                                                row = self._gru_pack[int(s)].detach().to('cpu').contiguous().float()
                                                gru_bank_pack_sha = hashlib.sha256(row.numpy().tobytes()).hexdigest()
                                        except Exception:
                                            gru_bank_pack_sha = ''
                                        try:
                                            if getattr(self, '_Wih', None) is not None and int(s) >= 0 and int(s) < int(getattr(self._Wih, 'shape', [0])[0]):
                                                row = self._Wih[int(s)].detach().to('cpu').contiguous().float()
                                                gru_bank_wih_sha = hashlib.sha256(row.numpy().tobytes()).hexdigest()
                                        except Exception:
                                            gru_bank_wih_sha = ''
                                        try:
                                            if getattr(self, '_Whh', None) is not None and int(s) >= 0 and int(s) < int(getattr(self._Whh, 'shape', [0])[0]):
                                                row = self._Whh[int(s)].detach().to('cpu').contiguous().float()
                                                gru_bank_whh_sha = hashlib.sha256(row.numpy().tobytes()).hexdigest()
                                        except Exception:
                                            gru_bank_whh_sha = ''
                                        try:
                                            wih_t = pol.gru.weight_ih.detach().cpu().contiguous().float()
                                            whh_t = pol.gru.weight_hh.detach().cpu().contiguous().float()
                                            bih_t = pol.gru.bias_ih.detach().cpu().contiguous().float().reshape(-1)
                                            bhh_t = pol.gru.bias_hh.detach().cpu().contiguous().float().reshape(-1)
                                            gru_expected_wih_sha = hashlib.sha256(wih_t.numpy().tobytes()).hexdigest()
                                            gru_expected_whh_sha = hashlib.sha256(whh_t.numpy().tobytes()).hexdigest()
                                            encW = pol.encoder.weight.detach().cpu().contiguous().float().reshape(-1)
                                            encb = pol.encoder.bias.detach().cpu().contiguous().float().reshape(-1)
                                            headW = pol.head.weight.detach().cpu().contiguous().float().reshape(-1)
                                            headb = pol.head.bias.detach().cpu().contiguous().float().reshape(-1)
                                            pack = torch.cat([encW, encb, wih_t.reshape(-1), bih_t, whh_t.reshape(-1), bhh_t, headW, headb], dim=0).contiguous().float()
                                            gru_expected_pack_sha = hashlib.sha256(pack.numpy().tobytes()).hexdigest()
                                        except Exception:
                                            pass
                                    elif pol is not None and getattr(pol, 'net', None) is not None:
                                        policy_kind = 'mlp'
                                except Exception:
                                    pass

                                last = last_seen.get(int(s), None)
                                added_details.append({
                                    'slot': int(s),
                                    'agent_id': aid,
                                    'slot_token': st,
                                    'bank_token': bt,
                                    'gru_bank_token': gbt,
                                    'mlp_bank_row_sha': bank_sha,
                                    'mlp_expected_row_sha': exp_sha,
                                    'mlp_row_mismatch': bool(bank_sha and exp_sha and bank_sha != exp_sha),
                                    'policy_kind': policy_kind,
                                    'gru_bank_pack_row_sha': gru_bank_pack_sha,
                                    'gru_expected_pack_row_sha': gru_expected_pack_sha,
                                    'gru_pack_row_mismatch': bool(gru_bank_pack_sha and gru_expected_pack_sha and gru_bank_pack_sha != gru_expected_pack_sha),
                                    'gru_bank_Wih_row_sha': gru_bank_wih_sha,
                                    'gru_expected_Wih_row_sha': gru_expected_wih_sha,
                                    'gru_Wih_row_mismatch': bool(gru_bank_wih_sha and gru_expected_wih_sha and gru_bank_wih_sha != gru_expected_wih_sha),
                                    'gru_bank_Whh_row_sha': gru_bank_whh_sha,
                                    'gru_expected_Whh_row_sha': gru_expected_whh_sha,
                                    'gru_Whh_row_mismatch': bool(gru_bank_whh_sha and gru_expected_whh_sha and gru_bank_whh_sha != gru_expected_whh_sha),
                                    'gru_hmem_row_sha': hmem_sha,
                                    'last_seen': last,
                                })
                            except Exception:
                                continue

                        # Update last-seen for currently alive slots
                        try:
                            for _a in _alive:
                                try:
                                    ss = int(getattr(_a, 'slot', -1))
                                    if ss < 0:
                                        continue
                                    last_seen[int(ss)] = {
                                        'tick': int(tick),
                                        'agent_id': int(getattr(_a, 'agent_id', -1)),
                                        'slot_token': int(getattr(_a, 'slot_token', 0)),
                                    }
                                except Exception:
                                    continue
                        except Exception:
                            pass

                        info['alive_slots_added_details'] = added_details
                        # Dirty-mark telemetry
                        info['dirty_mark_count'] = int(getattr(self, '_eiel_dirty_mark_count', 0) or 0)
                        try:
                            info['dirty_mark_events_tail'] = list(getattr(self, '_eiel_dirty_mark_events', []) or [])[-6:]
                        except Exception:
                            info['dirty_mark_events_tail'] = []

                        prev_map[int(generation_index)] = set(cur_set)
                        setattr(self, '_eiel_prev_alive_slots_for_bank_probe', prev_map)
                        setattr(self, '_eiel_slot_last_seen', last_seen)
            except Exception:
                pass

            emit_policy_bank_digest(run_dir, tag=tag, generation_index=generation_index, tick=tick, bank_info=info)
        except Exception:
            return

    def _eiel_emit_policy_bank_sync_probe(
        self,
        *,
        tag: str,
        generation_index: Optional[int],
        tick: Optional[int],
        kind: str,
        stage: str,
        sync_info: Dict[str, Any],
    ) -> None:
        """Emit a bank sync decision record to the run debug folder."""
        try:
            run_dir = str(getattr(self, 'run_dir', ''))
            if not run_dir:
                return
            emit_policy_bank_sync_digest(
                run_dir,
                tag=tag,
                generation_index=generation_index,
                tick=tick,
                kind=kind,
                stage=stage,
                sync_info=sync_info,
            )
        except Exception:
            return


    def _ensure_bank(self):
        if self._W1 is not None:
            return

        max_slots = int(getattr(self.pop, "max_slots", 0))
        if max_slots <= 0:
            # Resume safety:
            # If population.max_slots was not persisted/restored (e.g., legacy checkpoints
            # or early-gen diagnostics), infer a sufficient bank capacity from the
            # highest observed agent.slot to avoid index_select out-of-range.
            #
            # Must be deterministic and must not draw RNG.
            inferred = -1
            try:
                agents = getattr(self.pop, "agents", None)
                if isinstance(agents, dict) and agents:
                    for a in agents.values():
                        try:
                            s = int(getattr(a, "slot", -1))
                            if s > inferred:
                                inferred = s
                        except Exception:
                            continue
            except Exception:
                inferred = -1

            floor_cap = int(self.pop_cfg.population_size) * 2
            if inferred >= 0:
                max_slots = max(int(inferred) + 1, floor_cap)
            else:
                max_slots = floor_cap

        dev = self._bank_device
        dt = self._bank_dtype
        H = self._hidden
        O = self._obs_size
        A = self._out

        # MLP banks
        self._W1 = torch.zeros((max_slots, H, O), device=dev, dtype=dt)
        self._b1 = torch.zeros((max_slots, H), device=dev, dtype=dt)
        self._W2 = torch.zeros((max_slots, H, H), device=dev, dtype=dt)
        self._b2 = torch.zeros((max_slots, H), device=dev, dtype=dt)
        self._W3 = torch.zeros((max_slots, A, H), device=dev, dtype=dt)
        self._b3 = torch.zeros((max_slots, A), device=dev, dtype=dt)

        # Packed MLP bank (reduces per-tick gather overhead).
        # Layout per slot: [W1, b1, W2, b2, W3, b3] flattened and concatenated.
        P_mlp = (H * O) + H + (H * H) + H + (A * H) + A
        self._mlp_pack = torch.zeros((max_slots, P_mlp), device=dev, dtype=dt)

        # GRU banks are lazy-created when we see the first GRU agent
        self._gru_inited = False

        # IMPORTANT: fixed GRU hidden size for banked execution
        self._gru_hidden = 64

        self._Hmem = None  # (max_slots, Hm)
        self._encW = None
        self._encb = None
        self._Wih = None
        self._bih = None
        self._Whh = None
        self._bhh = None
        self._headW = None
        self._headb = None

    # -------------------------
    # GRU bank + memory state (FIXED SIZE)
    # -------------------------

    # -------------------------
    # Bank resize helpers (deterministic, no RNG)
    # -------------------------
    def _grow_mlp_bank(self, *, max_slots: int) -> None:
        """Grow MLP (and any initialized GRU) banks to at least `max_slots`.

        This is required for resume paths where restored agents may reference
        slot indices beyond the initially allocated bank capacity.

        Constraints:
        - Deterministic: no RNG draws.
        - Minimal: allocate larger tensors and copy existing rows.
        - No bundle schema changes.
        """
        self._ensure_bank()
        need = int(max_slots)
        cur = int(getattr(self._W1, 'shape', [0])[0]) if getattr(self, '_W1', None) is not None else 0
        if need <= cur:
            return

        dev = self._bank_device
        dt = self._bank_dtype
        H = int(self._hidden)
        O = int(self._obs_size)
        A = int(self._out)
        P_mlp = (H * O) + H + (H * H) + H + (A * H) + A

        def _grow_tensor(old_t: torch.Tensor, new_shape):
            new_t = torch.zeros(new_shape, device=dev, dtype=dt)
            if old_t is not None:
                n = min(int(old_t.shape[0]), int(new_shape[0]))
                if n > 0:
                    new_t[:n].copy_(old_t[:n])
            return new_t

        # MLP banks
        self._W1 = _grow_tensor(self._W1, (need, H, O))
        self._b1 = _grow_tensor(self._b1, (need, H))
        self._W2 = _grow_tensor(self._W2, (need, H, H))
        self._b2 = _grow_tensor(self._b2, (need, H))
        self._W3 = _grow_tensor(self._W3, (need, A, H))
        self._b3 = _grow_tensor(self._b3, (need, A))
        self._mlp_pack = _grow_tensor(getattr(self, '_mlp_pack', None), (need, P_mlp))

        # If GRU bank/memory was initialized, grow it too.
        if bool(getattr(self, '_gru_inited', False)):
            Hm = int(self._gru_hidden)
            self._Hmem = _grow_tensor(self._Hmem, (need, Hm))
            self._encW = _grow_tensor(self._encW, (need, Hm, O))
            self._encb = _grow_tensor(self._encb, (need, Hm))
            self._Wih = _grow_tensor(self._Wih, (need, 3 * Hm, Hm))
            self._bih = _grow_tensor(self._bih, (need, 3 * Hm))
            self._Whh = _grow_tensor(self._Whh, (need, 3 * Hm, Hm))
            self._bhh = _grow_tensor(self._bhh, (need, 3 * Hm))
            self._headW = _grow_tensor(self._headW, (need, A, Hm))
            self._headb = _grow_tensor(self._headb, (need, A))

        # Any resize invalidates previous clean state.
        self._mark_bank_dirty()

    def _ensure_gru_bank(self):
        self._ensure_bank()

        if self._gru_inited:
            return

        max_slots = int(self._W1.shape[0])
        dev = self._bank_device
        dt = self._bank_dtype
        O = self._obs_size
        A = self._out
        Hm = int(self._gru_hidden)

        # Memory state (persistent across steps, keyed by slot)
        self._Hmem = torch.zeros((max_slots, Hm), device=dev, dtype=dt)

        # Encoder: obs -> Hm
        self._encW = torch.zeros((max_slots, Hm, O), device=dev, dtype=dt)
        self._encb = torch.zeros((max_slots, Hm), device=dev, dtype=dt)

        # GRUCell parameters (per-slot)
        self._Wih = torch.zeros((max_slots, 3 * Hm, Hm), device=dev, dtype=dt)
        self._bih = torch.zeros((max_slots, 3 * Hm), device=dev, dtype=dt)
        self._Whh = torch.zeros((max_slots, 3 * Hm, Hm), device=dev, dtype=dt)
        self._bhh = torch.zeros((max_slots, 3 * Hm), device=dev, dtype=dt)

        # Head: Hm -> actions
        self._headW = torch.zeros((max_slots, A, Hm), device=dev, dtype=dt)
        self._headb = torch.zeros((max_slots, A), device=dev, dtype=dt)

        # Packed GRU bank (reduces per-tick gather overhead).
        # Layout per slot: [encW, encb, Wih, bih, Whh, bhh, headW, headb] flattened and concatenated.
        P_gru = (Hm * O) + Hm + (3 * Hm * Hm) + (3 * Hm) + (3 * Hm * Hm) + (3 * Hm) + (A * Hm) + A
        self._gru_pack = torch.zeros((max_slots, P_gru), device=dev, dtype=dt)

        self._gru_inited = True

    def _reset_memory_slots(self, slots: torch.Tensor):
        """
        Reset GRU memory state for the given slots (GPU tensor).
        Safe to call even if GRU bank isn't initialized.
        """
        if not getattr(self, "_gru_inited", False) or self._Hmem is None:
            return
        if slots.numel() == 0:
            return
        zeros = torch.zeros((slots.shape[0], self._gru_hidden), device=self._bank_device, dtype=self._bank_dtype)
        self._Hmem.index_copy_(0, slots, zeros)

    # -------------------------
    # MLP parameter bank sync
    # -------------------------
    def _sync_bank_for_agents(self, agents, buckets: Optional[Dict[str, float]] = None):
        """
        Sync ONLY MLP-like policies (must have `.net`).
        Safe to call with mixed cohorts; GRU agents will be ignored.
        """
        self._ensure_bank()
        if not agents:
            return
        # Resume-safety: ensure banks are large enough for the slot indices we are about to use.
        # This MUST run even if the bank is not marked dirty, because resume can restore agents
        # with slot indices beyond the initially allocated bank capacity.
        try:
            cur_cap = int(getattr(self._mlp_pack, 'shape', [0])[0]) if getattr(self, '_mlp_pack', None) is not None else 0
            max_slot = -1
            for a in agents:
                s = int(getattr(a, 'slot', -1))
                if s > max_slot:
                    max_slot = s
            if max_slot >= 0 and cur_cap > 0 and max_slot >= cur_cap:
                self._grow_mlp_bank(max_slots=int(max_slot) + 1)
        except Exception:
            # Never crash runs due to a resize guard.
            pass


        t0 = time.perf_counter()

        # ---- EIEL FDP: bank sync probe (Phase 3) ----
        def _env_truthy(key: str, default: str = "0") -> bool:
            v = str(os.getenv(key, default)).strip()
            return v in ("1", "true", "TRUE", "yes", "YES")

        # Enable by default when Forensic V4 is enabled (Phase 3 validation runs).
        dbg_sync = _env_truthy("EIEL_BANK_SYNC_PROBE", "0") or _env_truthy("EIEL_FORENSIC_V4", "0")
        _gen = None
        _tick = None
        _gen_runner = None
        _gen_pop = None
        _tick_runner = None
        _tick_pop = None

        _gen_fdp = None
        _tick_fdp = None
        _gen_attr = None
        _tick_attr = None
        try:
            _gen_fdp = int(getattr(self, "_eiel_fdp_cur_gen", -1))
        except Exception:
            _gen_fdp = None
        try:
            _tick_fdp = int(getattr(self, "_eiel_fdp_cur_tick", -1))
        except Exception:
            _tick_fdp = None
        try:
            _gen_attr = int(getattr(self, "gen", -1))
        except Exception:
            _gen_attr = None
        try:
            _tick_attr = int(getattr(self, "_tick", -1))
        except Exception:
            _tick_attr = None
        try:
            _gen_runner = int(getattr(self, "generation_index", -1))
        except Exception:
            _gen_runner = None
        try:
            _gen_pop = int(getattr(getattr(self, "pop", None), "generation_index", -1))
        except Exception:
            _gen_pop = None
        try:
            _tick_runner = int(getattr(self, "_tick", -1))
        except Exception:
            _tick_runner = None
        try:
            _tick_pop = int(getattr(getattr(self, "pop", None), "tick", -1))
        except Exception:
            _tick_pop = None

        _gen_fdp = None
        _tick_fdp = None
        _gen_attr = None
        _tick_attr = None
        try:
            _gen_fdp = int(getattr(self, "_eiel_fdp_cur_gen", -1))
        except Exception:
            _gen_fdp = None
        try:
            _tick_fdp = int(getattr(self, "_eiel_fdp_cur_tick", -1))
        except Exception:
            _tick_fdp = None
        try:
            _gen_attr = int(getattr(self, "gen", -1))
        except Exception:
            _gen_attr = None
        try:
            _tick_attr = int(getattr(self, "_tick", -1))
        except Exception:
            _tick_attr = None

        try:
            if _gen_fdp is not None and int(_gen_fdp) >= 0:
                _gen = int(_gen_fdp)
            elif _gen_attr is not None and int(_gen_attr) >= 0:
                _gen = int(_gen_attr)
            elif _gen_runner is not None and int(_gen_runner) >= 0:
                _gen = int(_gen_runner)
            elif _gen_pop is not None and int(_gen_pop) >= 0:
                _gen = int(_gen_pop)
            else:
                _gen = None
        except Exception:
            _gen = None

        try:
            if _tick_fdp is not None and int(_tick_fdp) >= 0:
                _tick = int(_tick_fdp)
            elif _tick_attr is not None and int(_tick_attr) >= 0:
                _tick = int(_tick_attr)
            elif _tick_runner is not None and int(_tick_runner) >= 0:
                _tick = int(_tick_runner)
            elif _tick_pop is not None and int(_tick_pop) >= 0:
                _tick = int(_tick_pop)
            else:
                _tick = None
        except Exception:
            _tick = None

        def _sha256_ints(vals: List[int]) -> str:
            try:
                b = ",".join(str(int(x)) for x in vals).encode("utf-8")
                return hashlib.sha256(b).hexdigest()
            except Exception:
                return ""

        def _sha256_ints_sorted(vals: List[int]) -> str:
            """Deterministic sha256 of the sorted integer list."""
            try:
                vv = sorted(int(x) for x in vals)
                b = ",".join(str(int(x)) for x in vv).encode("utf-8")
                return hashlib.sha256(b).hexdigest()
            except Exception:
                return ""

        def _mlp_pack_row_sha(slot: int) -> str:
            """Deterministic sha256 of the MLP packed row bytes for a given slot (float32 CPU)."""
            try:
                if getattr(self, "_mlp_pack", None) is None:
                    return ""
                ss = int(slot)
                if ss < 0:
                    return ""
                if ss >= int(getattr(self._mlp_pack, "shape", [0])[0]):
                    return ""
                row = self._mlp_pack[ss].detach()
                row_cpu = row.to("cpu").contiguous().float()
                return hashlib.sha256(row_cpu.numpy().tobytes()).hexdigest()
            except Exception:
                return ""

        def _want_sync_probe() -> bool:
            if not dbg_sync:
                return False
            if _gen != 3:
                return False
            if _tick is None or _tick < 0:
                return False
            try:
                s = os.getenv("EIEL_BANK_SYNC_PROBE_TICKS", "0,50,100,101,102,103,104,105,106,107,108,109,110,115,116")
                allow = set(int(x.strip()) for x in s.split(",") if x.strip() != "")
                return _tick in allow
            except Exception:
                return False

        _do_probe = _want_sync_probe()
        _tag = f"tick{int(_tick):03d}.bank_sync_mlp" if (_tick is not None and _tick >= 0) else "tick???.bank_sync_mlp"
        _cap0 = int(getattr(self._mlp_pack, 'shape', [0])[0]) if getattr(self, '_mlp_pack', None) is not None else 0
        if _do_probe:
            try:
                a_ids = [int(getattr(a, "agent_id", -1)) for a in agents]
                a_slots = [int(getattr(a, "slot", -1)) for a in agents]
                a_tokens = [int(getattr(a, "slot_token", 0)) for a in agents]
                sync_info0 = {
                    "n_agents_in": int(len(agents)),
                    "cap_mlp_pack": _cap0,
                    "agent_ids_sha256": _sha256_ints(a_ids),
                    "slots_sha256": _sha256_ints(a_slots),
                    "slot_tokens_sha256": _sha256_ints(a_tokens),
                    "tick_fdp": _tick_fdp,
                    "tick_attr": _tick_attr,
                    "tick_runner": _tick_runner,
                    "tick_pop": _tick_pop,
                    "gen_fdp": _gen_fdp,
                    "gen_attr": _gen_attr,
                    "gen_runner": _gen_runner,
                    "gen_pop": _gen_pop,
                    "bank_dirty_mlp": bool(getattr(self, "_bank_dirty_mlp", True)),
                }
                try:
                    # Small-cohort: include raw lists to debug per-call slot composition.
                    if int(len(agents)) <= 8:
                        sync_info0["agent_ids_list"] = list(a_ids)
                        sync_info0["slots_list"] = list(a_slots)
                        sync_info0["slot_tokens_list"] = list(a_tokens)
                except Exception:
                    pass
                self._eiel_emit_policy_bank_sync_probe(
                    tag=_tag, generation_index=_gen, tick=_tick,
                    kind="mlp", stage="enter", sync_info=sync_info0
                )
            except Exception:
                pass

        # Filter: only policies with `.net` (MLPPolicy)
        # NOTE: This must be defined *before* the skip-not-dirty path below.
        mlp_agents = [a for a in agents if hasattr(getattr(a, "policy", None), "net")]

        # Fast path: if nothing marked the bank dirty since last sync, skip.
        if not getattr(self, "_bank_dirty_mlp", True):

            if _do_probe:
                try:
                    self._eiel_emit_policy_bank_sync_probe(
                        tag=_tag, generation_index=_gen, tick=_tick,
                        kind="mlp", stage="skip_not_dirty", sync_info={"cap_mlp_pack": _cap0}
                    )
                except Exception:
                    pass

            # FDP v3: even when not dirty, check whether token gating or stale rows imply we SHOULD sync.
            if _do_probe:
                try:
                    if _gen == 3 and _tick is not None and (100 <= int(_tick) <= 110):
                        mlp_agents = [a for a in agents if hasattr(getattr(a, "policy", None), "net")]
                        mlp_ids = [int(getattr(a, "agent_id", -1)) for a in mlp_agents]
                        mlp_slots = [int(getattr(a, "slot", -1)) for a in mlp_agents]
                        mlp_tokens = [int(getattr(a, "slot_token", 0)) for a in mlp_agents]
                        mlp_bank_tokens = [int(getattr(a, "_bank_token", -1)) for a in mlp_agents]

                        # Would this cohort produce any to_sync due to token mismatch?
                        n_need_token = 0
                        for _tok, _last in zip(mlp_tokens, mlp_bank_tokens):
                            try:
                                if int(_tok) != int(_last):
                                    n_need_token += 1
                            except Exception:
                                continue

                        # Sample per-agent row hashes (bank vs expected) to detect stale inactive rows becoming active.
                        det = []
                        bank_row_sha = {}
                        exp_row_sha = {}
                        n_need_row = 0
                        # Stable sample order
                        _sample = list(mlp_agents)
                        try:
                            _sample.sort(key=lambda a: (int(getattr(a, "agent_id", -1)), int(getattr(a, "slot", -1))))
                        except Exception:
                            pass
                        _sample = _sample[:8]
                        for _a in _sample:
                            try:
                                aid = int(getattr(_a, "agent_id", -1))
                                ss = int(getattr(_a, "slot", -1))
                                st = int(getattr(_a, "slot_token", 0))
                                bt = int(getattr(_a, "_bank_token", -1))
                                bsha = _mlp_pack_row_sha(ss)
                                esha = ""
                                try:
                                    net = getattr(getattr(_a, "policy", None), "net", None)
                                    if net is not None:
                                        w1 = net[0].weight.detach().cpu().reshape(-1)
                                        b1 = net[0].bias.detach().cpu().reshape(-1)
                                        w2 = net[2].weight.detach().cpu().reshape(-1)
                                        b2 = net[2].bias.detach().cpu().reshape(-1)
                                        w3 = net[4].weight.detach().cpu().reshape(-1)
                                        b3 = net[4].bias.detach().cpu().reshape(-1)
                                        pack = torch.cat([w1, b1, w2, b2, w3, b3], dim=0).contiguous().float()
                                        esha = hashlib.sha256(pack.numpy().tobytes()).hexdigest()
                                except Exception:
                                    esha = ""
                                if ss >= 0:
                                    bank_row_sha[str(ss)] = bsha
                                    exp_row_sha[str(ss)] = esha
                                row_mismatch = bool(bsha and esha and (bsha != esha))
                                if row_mismatch:
                                    n_need_row += 1
                                det.append({
                                    "agent_id": aid,
                                    "slot": ss,
                                    "slot_token": st,
                                    "bank_token": bt,
                                    "token_mismatch": bool(st != bt),
                                    "bank_row_sha": bsha,
                                    "expected_row_sha": esha,
                                    "row_mismatch": row_mismatch,
                                })
                            except Exception:
                                continue

                        # Dirty-mark telemetry (callsite trace)
                        dirty_mark_count = int(getattr(self, "_eiel_dirty_mark_count", 0) or 0)
                        dirty_mark_tail = []
                        try:
                            _ev = list(getattr(self, "_eiel_dirty_mark_events", []) or [])
                            dirty_mark_tail = _ev[-6:]
                        except Exception:
                            dirty_mark_tail = []

                        self._eiel_emit_policy_bank_sync_probe(
                            tag=_tag, generation_index=_gen, tick=_tick,
                            kind="mlp", stage="skip_not_dirty_details",
                            sync_info={
                                "n_mlp": int(len(mlp_agents)),
                                "token_mismatch_count": int(n_need_token),
                                "row_mismatch_count_sample": int(n_need_row),
                                "mlp_agent_ids_sha256_raw": _sha256_ints(mlp_ids),
                                "mlp_agent_ids_sha256_sorted": _sha256_ints_sorted(mlp_ids),
                                "mlp_slots_sha256_raw": _sha256_ints(mlp_slots),
                                "mlp_slots_sha256_sorted": _sha256_ints_sorted(mlp_slots),
                                "mlp_slot_tokens_sha256_raw": _sha256_ints(mlp_tokens),
                                "mlp_slot_tokens_sha256_sorted": _sha256_ints_sorted(mlp_tokens),
                                "mlp_bank_tokens_sha256_raw": _sha256_ints(mlp_bank_tokens),
                                "mlp_bank_tokens_sha256_sorted": _sha256_ints_sorted(mlp_bank_tokens),
                                "mlp_pack_row_sha_sample_by_slot": bank_row_sha,
                                "expected_mlp_row_sha_sample_by_slot": exp_row_sha,
                                "details_sample": det,
                                "dirty_mark_count": dirty_mark_count,
                                "dirty_mark_events_tail": dirty_mark_tail,
                            }
                        )
                except Exception:
                    pass


            # Determinism guard: token mismatch must force sync even if bank isn't marked dirty.
            # Rationale: slot_token!=bank_token implies slot reassignment or resume gap; skipping leaves stale rows.
            force_full = False
            try:
                for _a in mlp_agents:
                    if bool(getattr(_a, '_eiel_force_full_mlp_bank_sync_once', False)):
                        force_full = True
                        break
            except Exception:
                force_full = False
            n_need_token = 0
            mismatch_slots = []
            try:
                for _a in mlp_agents:
                    try:
                        _slot = int(getattr(_a, 'slot', -1))
                        if _slot < 0:
                            continue
                        _tok = int(getattr(_a, 'slot_token', 0))
                        _last = int(getattr(_a, '_bank_token', -1))
                        if _tok != _last:
                            n_need_token += 1
                            mismatch_slots.append(_slot)
                    except Exception:
                        continue
            except Exception:
                n_need_token = 0
            if (not force_full) and n_need_token == 0:
                if buckets is not None:
                    buckets["bank_sync"] = buckets.get("bank_sync", 0.0) + (time.perf_counter() - t0)
                return
            else:
                if _do_probe:
                    try:
                        self._eiel_emit_policy_bank_sync_probe(
                            tag=_tag, generation_index=_gen, tick=_tick,
                            kind='mlp', stage='override_token_mismatch',
                            sync_info={'n_need_token': int(n_need_token), 'force_full': bool(force_full)}
                        )
                    except Exception:
                        pass
                # If slots were reassigned (token mismatch), clear irrelevant GRU bank rows deterministically.
                try:
                    if mismatch_slots and (
                        getattr(self, "_gru_pack", None) is not None
                        or getattr(self, "_gru_Hmem", None) is not None
                        or getattr(self, "_gru_Wih", None) is not None
                        or getattr(self, "_gru_Whh", None) is not None
                    ):
                        self._ensure_gru_bank()
                        _uniq = sorted({int(s) for s in mismatch_slots if int(s) >= 0})
                        if _uniq:
                            _idx = torch.tensor(_uniq, dtype=torch.long, device=self._gru_pack.device)
                            for _tname in ("_gru_pack", "_gru_Wih", "_gru_Whh", "_gru_Hmem"):
                                _t = getattr(self, _tname, None)
                                if _t is None:
                                    continue
                                if _t.shape[0] > 0:
                                    _t.index_fill_(0, _idx, 0.0)
                            if _do_probe:
                                self._eiel_emit_policy_bank_sync_probe(
                                    tag=_tag, generation_index=_gen, tick=_tick,
                                    kind='mlp', stage='clear_irrelevant_rows',
                                    sync_info={'n_slots': int(len(_uniq))}
                                )
                except Exception:
                    pass

                # If slots were reassigned (token mismatch), clear irrelevant MLP bank rows deterministically.
                try:
                    if mismatch_slots:
                        # Ensure bank tensors exist (previous versions called a
                        # non-existent helper and silently swallowed the error).
                        self._ensure_bank()
                        _uniq = sorted({int(s) for s in mismatch_slots if int(s) >= 0})
                        if _uniq:
                            _idx = torch.tensor(_uniq, dtype=torch.long, device=self._mlp_pack.device)
                            if self._mlp_pack is not None and self._mlp_pack.shape[0] > 0:
                                self._mlp_pack.index_fill_(0, _idx, 0.0)
                            if _do_probe:
                                self._eiel_emit_policy_bank_sync_probe(
                                    tag=_tag, generation_index=_gen, tick=_tick,
                                    kind='mlp', stage='clear_irrelevant_rows',
                                    sync_info={'n_slots': int(len(_uniq))}
                                )
                except Exception:
                    pass

                # Fall through into normal sync path below (compute to_sync and apply).

        # Filter: only policies with `.net` (MLPPolicy)
        mlp_agents = [a for a in agents if hasattr(getattr(a, "policy", None), "net")]
        if not mlp_agents:
            if buckets is not None:
                buckets["bank_sync"] = buckets.get("bank_sync", 0.0) + (time.perf_counter() - t0)
            return

        # IMPORTANT (Phase 3/4 determinism): a dirty bank means "content unknown".
        # Even if per-agent slot_token appears to match `_bank_token`, we must do
        # a full sync at least once to ensure the bank reflects current weights.
        # This prevents resume-time fast-path skipping from leaving stale rows.
        force_full = bool(getattr(self, "_bank_dirty_mlp", True))
        for a in mlp_agents:
            if bool(getattr(a, '_eiel_force_full_mlp_bank_sync_once', False)):
                force_full = True
                break

        to_sync = []
        for a in mlp_agents:
            slot = int(getattr(a, "slot", -1))
            if slot < 0:
                continue
            if force_full:
                to_sync.append(a)
                continue
            token = int(getattr(a, "slot_token", 0))
            last = int(getattr(a, "_bank_token", -1))
            if token != last:
                to_sync.append(a)

        if not to_sync:

            if _do_probe:
                try:
                    # token gating summary
                    n_mlp = int(len(mlp_agents))
                    n_need = 0
                    for _a in mlp_agents:
                        try:
                            tok = int(getattr(_a, "slot_token", 0))
                            last = int(getattr(_a, "_bank_token", -1))
                            if tok != last:
                                n_need += 1
                        except Exception:
                            continue
                    self._eiel_emit_policy_bank_sync_probe(
                        tag=_tag, generation_index=_gen, tick=_tick,
                        kind="mlp", stage="skip_no_to_sync",
                        sync_info={"n_mlp": n_mlp, "n_need": n_need, "cap_mlp_pack": _cap0, "force_full": bool(force_full)}
                    )
                except Exception:
                    pass
            if buckets is not None:
                buckets["bank_sync"] = buckets.get("bank_sync", 0.0) + (time.perf_counter() - t0)
            return

        
        if _do_probe:
            try:
                ts_ids = [int(getattr(a, "agent_id", -1)) for a in to_sync]
                ts_slots = [int(getattr(a, "slot", -1)) for a in to_sync]
                ts_tokens = [int(getattr(a, "slot_token", 0)) for a in to_sync]
                self._eiel_emit_policy_bank_sync_probe(
                    tag=_tag, generation_index=_gen, tick=_tick,
                    kind="mlp", stage="pre_apply",
                    sync_info={
                        "n_to_sync": int(len(to_sync)),
                        "to_sync_agent_ids_sha256": _sha256_ints(ts_ids),
                        "to_sync_slots_sha256": _sha256_ints(ts_slots),
                        "to_sync_slot_tokens_sha256": _sha256_ints(ts_tokens),
                        "cap_mlp_pack": _cap0,
                        "force_full": bool(force_full),
                    }
                )
            except Exception:
                pass
        if _do_probe and int(_tick) >= 100:
            _slot_agent_pairs_sha256_by_slot = ""
            try:
                mlp_ids = [int(getattr(a, "agent_id", -1)) for a in mlp_agents]
                mlp_slots = [int(getattr(a, "slot", -1)) for a in mlp_agents]
                mlp_tokens = [int(getattr(a, "slot_token", 0)) for a in mlp_agents]
                ts_ids = [int(getattr(a, "agent_id", -1)) for a in to_sync]
                ts_slots = [int(getattr(a, "slot", -1)) for a in to_sync]
                ts_tokens = [int(getattr(a, "slot_token", 0)) for a in to_sync]

                def _dupes(vals: List[int]) -> List[int]:
                    seen = set()
                    dup = set()
                    for v in vals:
                        vv = int(v)
                        if vv in seen:
                            dup.add(vv)
                        seen.add(vv)
                    return sorted(int(x) for x in dup if int(x) >= 0)[:16]

                to_sync_set = set(int(getattr(a, "agent_id", -1)) for a in to_sync)
                det = []
                for _a in mlp_agents:
                    try:
                        aid = int(getattr(_a, "agent_id", -1))
                        det.append({
                            "agent_id": aid,
                            "slot": int(getattr(_a, "slot", -1)),
                            "slot_token": int(getattr(_a, "slot_token", 0)),
                            "bank_token": int(getattr(_a, "_bank_token", -1)),
                            "in_to_sync": bool(aid in to_sync_set),
                            "force_full_once": bool(getattr(_a, "_eiel_force_full_mlp_bank_sync_once", False)),
                        })
                    except Exception:
                        continue
                det.sort(key=lambda d: (int(d.get("agent_id", -1)), int(d.get("slot", -1))))
                try:
                    _pairs = [(int(d.get("slot", -1)), int(d.get("agent_id", -1))) for d in det if int(d.get("slot", -1)) >= 0]
                    _pairs.sort(key=lambda p: (int(p[0]), int(p[1])))
                    _pairs_b = ";".join(f"{int(s)}:{int(a)}" for (s, a) in _pairs).encode("utf-8")
                    _slot_agent_pairs_sha256_by_slot = hashlib.sha256(_pairs_b).hexdigest()
                except Exception:
                    _slot_agent_pairs_sha256_by_slot = ""


                pack_pre = {}
                for d in det:
                    try:
                        ss = int(d.get("slot", -1))
                        if ss >= 0:
                            pack_pre[str(ss)] = _mlp_pack_row_sha(ss)
                    except Exception:
                        continue

                n_token_mismatch = 0
                for d in det:
                    try:
                        if int(d.get("slot_token", 0)) != int(d.get("bank_token", -1)):
                            n_token_mismatch += 1
                    except Exception:
                        continue

                self._eiel_emit_policy_bank_sync_probe(
                    tag=_tag, generation_index=_gen, tick=_tick,
                    kind="mlp", stage=f"tick{int(_tick):03d}_pre_apply_details",
                    sync_info={
                        "n_mlp": int(len(mlp_agents)),
                        "n_to_sync": int(len(to_sync)),
                        "mlp_agent_ids_sha256_raw": _sha256_ints(mlp_ids),
                        "mlp_agent_ids_sha256_sorted": _sha256_ints_sorted(mlp_ids),
                        "mlp_slots_sha256_raw": _sha256_ints(mlp_slots),
                        "mlp_slots_sha256_sorted": _sha256_ints_sorted(mlp_slots),
                        "mlp_slot_tokens_sha256_raw": _sha256_ints(mlp_tokens),
                        "mlp_slot_tokens_sha256_sorted": _sha256_ints_sorted(mlp_tokens),
                        "to_sync_agent_ids_sha256_raw": _sha256_ints(ts_ids),
                        "to_sync_agent_ids_sha256_sorted": _sha256_ints_sorted(ts_ids),
                        "to_sync_slots_sha256_raw": _sha256_ints(ts_slots),
                        "to_sync_slots_sha256_sorted": _sha256_ints_sorted(ts_slots),
                        "to_sync_slot_tokens_sha256_raw": _sha256_ints(ts_tokens),
                        "to_sync_slot_tokens_sha256_sorted": _sha256_ints_sorted(ts_tokens),
                        "dupe_slots_in_mlp_agents": _dupes(mlp_slots),
                        "dupe_slots_in_to_sync": _dupes(ts_slots),
                        "token_mismatch_count": int(n_token_mismatch),
                        "slot_agent_pairs_sha256_by_slot": _slot_agent_pairs_sha256_by_slot,
                        "mlp_pack_row_sha_pre_by_slot": pack_pre,
                        "details": det,
                    }
                )
            except Exception:
                pass

        slots = torch.tensor([int(a.slot) for a in to_sync], device="cpu", dtype=torch.long)

        w1 = torch.stack([a.policy.net[0].weight.detach().cpu() for a in to_sync], dim=0)
        b1 = torch.stack([a.policy.net[0].bias.detach().cpu().reshape(-1) for a in to_sync], dim=0)
        w2 = torch.stack([a.policy.net[2].weight.detach().cpu() for a in to_sync], dim=0)
        b2 = torch.stack([a.policy.net[2].bias.detach().cpu().reshape(-1) for a in to_sync], dim=0)
        w3 = torch.stack([a.policy.net[4].weight.detach().cpu() for a in to_sync], dim=0)
        b3 = torch.stack([a.policy.net[4].bias.detach().cpu().reshape(-1) for a in to_sync], dim=0)

        # Build packed parameter rows on CPU, then a single index_copy_ to GPU.
        # This is intentionally redundant with the individual tensors; those remain for compatibility.
        pack_cpu = torch.cat([
            w1.reshape(w1.shape[0], -1),
            b1.reshape(b1.shape[0], -1),
            w2.reshape(w2.shape[0], -1),
            b2.reshape(b2.shape[0], -1),
            w3.reshape(w3.shape[0], -1),
            b3.reshape(b3.shape[0], -1),
        ], dim=1)

        slots_gpu = slots.to(self._bank_device, non_blocking=True)

        # Single packed copy (main throughput win).
        if getattr(self, "_mlp_pack", None) is not None:
            self._mlp_pack.index_copy_(0, slots_gpu, pack_cpu.to(self._bank_device, non_blocking=True))

        self._W1.index_copy_(0, slots_gpu, w1.to(self._bank_device, non_blocking=True))
        self._b1.index_copy_(0, slots_gpu, b1.to(self._bank_device, non_blocking=True))
        self._W2.index_copy_(0, slots_gpu, w2.to(self._bank_device, non_blocking=True))
        self._b2.index_copy_(0, slots_gpu, b2.to(self._bank_device, non_blocking=True))
        self._W3.index_copy_(0, slots_gpu, w3.to(self._bank_device, non_blocking=True))
        self._b3.index_copy_(0, slots_gpu, b3.to(self._bank_device, non_blocking=True))

        for a in to_sync:
            a._bank_token = int(a.slot_token)
            if force_full:
                try:
                    a._eiel_force_full_mlp_bank_sync_once = False
                except Exception:
                    pass

        # Bank is now consistent for the current cohort.
        self._bank_dirty_mlp = False

        if _do_probe and int(_tick) >= 100:
            _slot_agent_pairs_sha256_by_slot = ""
            try:
                mlp_slots = [int(getattr(a, "slot", -1)) for a in mlp_agents]
                to_sync_set = set(int(getattr(a, "agent_id", -1)) for a in to_sync)
                det = []
                for _a in mlp_agents:
                    try:
                        aid = int(getattr(_a, "agent_id", -1))
                        det.append({
                            "agent_id": aid,
                            "slot": int(getattr(_a, "slot", -1)),
                            "slot_token": int(getattr(_a, "slot_token", 0)),
                            "bank_token": int(getattr(_a, "_bank_token", -1)),
                            "in_to_sync": bool(aid in to_sync_set),
                        })
                    except Exception:
                        continue
                det.sort(key=lambda d: (int(d.get("agent_id", -1)), int(d.get("slot", -1))))

                pack_post = {}
                for d in det:
                    try:
                        ss = int(d.get("slot", -1))
                        if ss >= 0:
                            pack_post[str(ss)] = _mlp_pack_row_sha(ss)
                    except Exception:
                        continue

                n_token_mismatch = 0
                for d in det:
                    try:
                        if int(d.get("slot_token", 0)) != int(d.get("bank_token", -1)):
                            n_token_mismatch += 1
                    except Exception:
                        continue

                self._eiel_emit_policy_bank_sync_probe(
                    tag=_tag, generation_index=_gen, tick=_tick,
                    kind="mlp", stage=f"tick{int(_tick):03d}_post_apply_details",
                    sync_info={
                        "n_mlp": int(len(mlp_agents)),
                        "n_to_sync": int(len(to_sync)),
                        "mlp_slots_sha256_sorted": _sha256_ints_sorted(mlp_slots),
                        "token_mismatch_count": int(n_token_mismatch),
                        "slot_agent_pairs_sha256_by_slot": _slot_agent_pairs_sha256_by_slot,
                        "mlp_pack_row_sha_post_by_slot": pack_post,
                        "details": det,
                    }
                )
            except Exception:
                pass

        if _do_probe:
            try:
                cap1 = int(getattr(self._mlp_pack, 'shape', [0])[0]) if getattr(self, '_mlp_pack', None) is not None else 0
                self._eiel_emit_policy_bank_sync_probe(
                    tag=_tag, generation_index=_gen, tick=_tick,
                    kind="mlp", stage="post_apply",
                    sync_info={"cap_mlp_pack": cap1, "n_to_sync": int(len(to_sync)), "force_full": bool(force_full)}
                )
            except Exception:
                pass

        if buckets is not None:
            buckets["bank_sync"] = buckets.get("bank_sync", 0.0) + (time.perf_counter() - t0)

    # -------------------------
    # GRU parameter bank sync (FIXED SIZE)
    # -------------------------
    def _sync_gru_bank_for_agents(self, agents, buckets: Optional[Dict[str, float]] = None):
        """
        Sync ONLY GRU-like policies (must have `.gru`, `.encoder`, `.head`).
        Assumes all GRU policies are built with hidden_dim == self._gru_hidden (64).
        """
        if not agents:
            return

        # Filter: only policies with `.gru` (GRUPolicy)
        gru_agents = [a for a in agents if hasattr(getattr(a, "policy", None), "gru")]
        if not gru_agents:
            return

        self._ensure_gru_bank()

        # Validate fixed size: if any GRU policy isn't 64, fail loud (better than silent bugs)
        Hm = int(self._gru_hidden)
        first = gru_agents[0]
        pol_h = int(getattr(first.policy, "hidden_dim", 0))
        if pol_h <= 0:
            pol_h = int(first.policy.gru.weight_hh.shape[1])
        if int(pol_h) != Hm:
            raise ValueError(
                f"GRU policy hidden_dim={pol_h} but bank fixed hidden_dim={Hm}. Rebuild GRU policies to 64."
            )

        t0 = time.perf_counter()

        # ---- EIEL FDP: bank sync probe (Phase 3) ----
        def _env_truthy(key: str, default: str = "0") -> bool:
            v = str(os.getenv(key, default)).strip()
            return v in ("1", "true", "TRUE", "yes", "YES")

        # Enable by default when Forensic V4 is enabled (Phase 3 validation runs).
        dbg_sync = _env_truthy("EIEL_BANK_SYNC_PROBE", "0") or _env_truthy("EIEL_FORENSIC_V4", "0")
        _gen = None
        _tick = None
        _gen_runner = None
        _gen_pop = None
        _tick_runner = None
        _tick_pop = None
        try:
            _gen_runner = int(getattr(self, "generation_index", -1))
        except Exception:
            _gen_runner = None
        try:
            _gen_pop = int(getattr(getattr(self, "pop", None), "generation_index", -1))
        except Exception:
            _gen_pop = None
        try:
            _tick_runner = int(getattr(self, "_tick", -1))
        except Exception:
            _tick_runner = None
        try:
            _tick_pop = int(getattr(getattr(self, "pop", None), "tick", -1))
        except Exception:
            _tick_pop = None
        _gen_fdp = None
        _tick_fdp = None
        _gen_attr = None
        _tick_attr = None
        try:
            _gen_fdp = int(getattr(self, "_eiel_fdp_cur_gen", -1))
        except Exception:
            _gen_fdp = None
        try:
            _tick_fdp = int(getattr(self, "_eiel_fdp_cur_tick", -1))
        except Exception:
            _tick_fdp = None
        try:
            _gen_attr = int(getattr(self, "gen", -1))
        except Exception:
            _gen_attr = None
        try:
            _tick_attr = int(getattr(self, "_tick", -1))
        except Exception:
            _tick_attr = None


        try:
            if _gen_fdp is not None and int(_gen_fdp) >= 0:
                _gen = int(_gen_fdp)
            elif _gen_attr is not None and int(_gen_attr) >= 0:
                _gen = int(_gen_attr)
            elif _gen_runner is not None and int(_gen_runner) >= 0:
                _gen = int(_gen_runner)
            elif _gen_pop is not None and int(_gen_pop) >= 0:
                _gen = int(_gen_pop)
            else:
                _gen = None
        except Exception:
            _gen = None

        try:
            if _tick_fdp is not None and int(_tick_fdp) >= 0:
                _tick = int(_tick_fdp)
            elif _tick_attr is not None and int(_tick_attr) >= 0:
                _tick = int(_tick_attr)
            elif _tick_runner is not None and int(_tick_runner) >= 0:
                _tick = int(_tick_runner)
            elif _tick_pop is not None and int(_tick_pop) >= 0:
                _tick = int(_tick_pop)
            else:
                _tick = None
        except Exception:
            _tick = None

        def _sha256_ints(vals: List[int]) -> str:
            try:
                b = ",".join(str(int(x)) for x in vals).encode("utf-8")
                return hashlib.sha256(b).hexdigest()
            except Exception:
                return ""

        def _sha256_ints_sorted(vals: List[int]) -> str:
            """Deterministic sha256 of the sorted integer list."""
            try:
                vv = sorted(int(x) for x in vals)
                b = ",".join(str(int(x)) for x in vv).encode("utf-8")
                return hashlib.sha256(b).hexdigest()
            except Exception:
                return ""

        def _hmem_row_sha(slot: int) -> str:
            """Deterministic sha256 of Hmem row bytes for a given slot (float32 CPU)."""
            try:
                if not getattr(self, "_gru_inited", False) or getattr(self, "_Hmem", None) is None:
                    return ""
                if slot < 0:
                    return ""
                # Force a stable CPU float32 view before hashing.
                row = self._Hmem[int(slot)].detach()
                row_cpu = row.to("cpu").contiguous().float()
                return hashlib.sha256(row_cpu.numpy().tobytes()).hexdigest()
            except Exception:
                return ""

        def _hmem_slots_sha(slots_list: List[int]) -> Dict[str, str]:
            """Map slot->sha256(Hmem_row) for the given slots (as strings for JSON stability)."""
            out: Dict[str, str] = {}
            try:
                for s in slots_list:
                    ss = int(s)
                    out[str(ss)] = _hmem_row_sha(ss)
            except Exception:
                pass
            return out

        def _tensor_slot_sha(t: Optional[torch.Tensor], slot: int) -> str:
            """Deterministic sha256 of tensor[slot] bytes (float32 CPU)."""
            try:
                if t is None:
                    return ""
                ss = int(slot)
                if ss < 0:
                    return ""
                if ss >= int(getattr(t, "shape", [0])[0]):
                    return ""
                row = t[ss].detach()
                row_cpu = row.to("cpu").contiguous().float()
                return hashlib.sha256(row_cpu.numpy().tobytes()).hexdigest()
            except Exception:
                return ""

        def _gru_rows_sha(slots_list: List[int]) -> Dict[str, Dict[str, str]]:
            """Map slot-> {gru_pack,Wih,Whh,Hmem} sha256 for the given slots."""
            out: Dict[str, Dict[str, str]] = {}
            try:
                pack = getattr(self, "_gru_pack", None)
                for s in slots_list:
                    ss = int(s)
                    if ss < 0:
                        continue
                    out[str(ss)] = {
                        "gru_pack": _tensor_slot_sha(pack, ss),
                        "Wih": _tensor_slot_sha(getattr(self, "_Wih", None), ss),
                        "Whh": _tensor_slot_sha(getattr(self, "_Whh", None), ss),
                        "Hmem": _tensor_slot_sha(getattr(self, "_Hmem", None), ss),
                    }
            except Exception:
                pass
            return out

        def _dupes(vals: List[int]) -> List[int]:
            """Return up to 16 duplicate (>=0) ints from the list, sorted."""
            seen = set()
            dup = set()
            for v in vals:
                vv = int(v)
                if vv in seen:
                    dup.add(vv)
                seen.add(vv)
            return sorted(int(x) for x in dup if int(x) >= 0)[:16]

        def _want_sync_probe() -> bool:
            if not dbg_sync:
                return False
            if _gen != 3:
                return False
            if _tick is None or _tick < 0:
                return False
            try:
                s = os.getenv("EIEL_BANK_SYNC_PROBE_TICKS", "0,50,100,101,102,103,104,105,106,107,108,109,110,115,116")
                allow = set(int(x.strip()) for x in s.split(",") if x.strip() != "")
                return _tick in allow
            except Exception:
                return False

        _do_probe = _want_sync_probe()
        _tag = f"tick{int(_tick):03d}.bank_sync_gru" if (_tick is not None and _tick >= 0) else "tick???.bank_sync_gru"
        _cap0 = int(getattr(self._gru_pack, 'shape', [0])[0]) if getattr(self, '_gru_pack', None) is not None else 0
        if _do_probe:
            try:
                a_ids = [int(getattr(a, "agent_id", -1)) for a in agents]
                a_slots = [int(getattr(a, "slot", -1)) for a in agents]
                a_tokens = [int(getattr(a, "slot_token", 0)) for a in agents]
                sync_info0 = {
                    "n_agents_in": int(len(agents)),
                    "cap_gru_pack": _cap0,
                    "agent_ids_sha256": _sha256_ints(a_ids),
                    "agent_ids_sha256_sorted": _sha256_ints_sorted(a_ids),
                    "agent_ids_sha256_raw": _sha256_ints(a_ids),
                    "slots_sha256_sorted": _sha256_ints_sorted(a_slots),
                    "slots_sha256_raw": _sha256_ints(a_slots),
                    "slot_tokens_sha256_sorted": _sha256_ints_sorted(a_tokens),
                    "slot_tokens_sha256_raw": _sha256_ints(a_tokens),
                    "slots_sha256": _sha256_ints(a_slots),
                    "slot_tokens_sha256": _sha256_ints(a_tokens),
                    "tick_fdp": _tick_fdp,
                    "tick_attr": _tick_attr,
                    "tick_runner": _tick_runner,
                    "tick_pop": _tick_pop,
                    "gen_fdp": _gen_fdp,
                    "gen_attr": _gen_attr,
                    "gen_runner": _gen_runner,
                    "gen_pop": _gen_pop,
                    "bank_dirty_gru": bool(getattr(self, "_bank_dirty_gru", True)),
                    "skip_gru_memory_reset_once": bool(getattr(self, "_eiel_skip_gru_memory_reset_once", False)),
                }
                if int(_tick) == 110:
                    try:
                        det = []
                        for _a in agents:
                            try:
                                det.append({
                                    "agent_id": int(getattr(_a, "agent_id", -1)),
                                    "slot": int(getattr(_a, "slot", -1)),
                                    "slot_token": int(getattr(_a, "slot_token", 0)),
                                    "gru_bank_token": int(getattr(_a, "_gru_bank_token", -1)),
                                    "force_full_once": bool(getattr(_a, "_eiel_force_full_gru_bank_sync_once", False)),
                                })
                            except Exception:
                                continue
                        # Stable ordering for diff: by agent_id then slot.
                        det.sort(key=lambda d: (int(d.get("agent_id", -1)), int(d.get("slot", -1))))
                        sync_info0["tick110_agent_details"] = det
                        # Hmem row hashes for involved slots (stable keys).
                        sync_info0["tick110_hmem_by_slot_sha_pre"] = _hmem_slots_sha([int(d.get("slot",-1)) for d in det if int(d.get("slot",-1)) >= 0])
                    except Exception:
                        pass
                self._eiel_emit_policy_bank_sync_probe(
                    tag=_tag, generation_index=_gen, tick=_tick,
                    kind="gru", stage="enter", sync_info=sync_info0
                )
            except Exception:
                pass

        # Fast path: if nothing marked the GRU bank dirty since last sync, skip.
        if not getattr(self, "_bank_dirty_gru", True):

            if _do_probe:
                try:
                    self._eiel_emit_policy_bank_sync_probe(
                        tag=_tag, generation_index=_gen, tick=_tick,
                        kind="gru", stage="skip_not_dirty", sync_info={"cap_gru_pack": _cap0}
                    )
                except Exception:
                    pass

            # FDP v3: even when not dirty, check whether token gating or stale rows imply we SHOULD sync.
            if _do_probe:
                try:
                    if _gen == 3 and _tick is not None and (100 <= int(_tick) <= 110):
                        gru_ids = [int(getattr(a, "agent_id", -1)) for a in gru_agents]
                        gru_slots = [int(getattr(a, "slot", -1)) for a in gru_agents]
                        gru_tokens = [int(getattr(a, "slot_token", 0)) for a in gru_agents]
                        gru_bank_tokens = [int(getattr(a, "_gru_bank_token", -1)) for a in gru_agents]

                        n_need_token = 0
                        for _tok, _last in zip(gru_tokens, gru_bank_tokens):
                            try:
                                if int(_tok) != int(_last):
                                    n_need_token += 1
                            except Exception:
                                continue

                        det = []
                        hmem_row_sha = {}
                        gru_pack_row_sha = {}
                        gru_wih_row_sha = {}
                        gru_whh_row_sha = {}
                        n_hmem_nonzero = 0
                        _sample = list(gru_agents)
                        try:
                            _sample.sort(key=lambda a: (int(getattr(a, "agent_id", -1)), int(getattr(a, "slot", -1))))
                        except Exception:
                            pass
                        _sample = _sample[:8]
                        for _a in _sample:
                            try:
                                aid = int(getattr(_a, "agent_id", -1))
                                ss = int(getattr(_a, "slot", -1))
                                st = int(getattr(_a, "slot_token", 0))
                                bt = int(getattr(_a, "_gru_bank_token", -1))
                                bsha = ""
                                try:
                                    if getattr(self, "_Hmem", None) is not None and ss >= 0 and ss < int(getattr(self._Hmem, "shape", [0])[0]):
                                        row = self._Hmem[ss].detach().to("cpu").contiguous().float()
                                        bsha = hashlib.sha256(row.numpy().tobytes()).hexdigest()
                                        # crude nonzero check
                                        try:
                                            if float(row.abs().sum().item()) > 0.0:
                                                n_hmem_nonzero += 1
                                        except Exception:
                                            pass
                                except Exception:
                                    bsha = ""
                                pack_sha = ''
                                wih_sha = ''
                                whh_sha = ''
                                exp_pack_sha = ''
                                exp_wih_sha = ''
                                exp_whh_sha = ''
                                try:
                                    pack_sha = _tensor_slot_sha(getattr(self, '_gru_pack', None), ss)
                                    wih_sha = _tensor_slot_sha(getattr(self, '_Wih', None), ss)
                                    whh_sha = _tensor_slot_sha(getattr(self, '_Whh', None), ss)
                                except Exception:
                                    pass
                                try:
                                    pol = getattr(_a, 'policy', None)
                                    if pol is not None and getattr(pol, 'gru', None) is not None:
                                        wih_t = pol.gru.weight_ih.detach().cpu().contiguous().float()
                                        whh_t = pol.gru.weight_hh.detach().cpu().contiguous().float()
                                        bih_t = pol.gru.bias_ih.detach().cpu().contiguous().float().reshape(-1)
                                        bhh_t = pol.gru.bias_hh.detach().cpu().contiguous().float().reshape(-1)
                                        exp_wih_sha = hashlib.sha256(wih_t.numpy().tobytes()).hexdigest()
                                        exp_whh_sha = hashlib.sha256(whh_t.numpy().tobytes()).hexdigest()
                                        encW = pol.encoder.weight.detach().cpu().contiguous().float().reshape(-1)
                                        encb = pol.encoder.bias.detach().cpu().contiguous().float().reshape(-1)
                                        headW = pol.head.weight.detach().cpu().contiguous().float().reshape(-1)
                                        headb = pol.head.bias.detach().cpu().contiguous().float().reshape(-1)
                                        pack = torch.cat([encW, encb, wih_t.reshape(-1), bih_t, whh_t.reshape(-1), bhh_t, headW, headb], dim=0).contiguous().float()
                                        exp_pack_sha = hashlib.sha256(pack.numpy().tobytes()).hexdigest()
                                except Exception:
                                    pass
                                if ss >= 0:
                                    hmem_row_sha[str(ss)] = bsha
                                    gru_pack_row_sha[str(ss)] = pack_sha
                                    gru_wih_row_sha[str(ss)] = wih_sha
                                    gru_whh_row_sha[str(ss)] = whh_sha
                                det.append({
                                    "agent_id": aid,
                                    "slot": ss,
                                    "slot_token": st,
                                    "gru_bank_token": bt,
                                    "token_mismatch": bool(st != bt),
                                    "hmem_row_sha": bsha,
                                    "gru_pack_row_sha": pack_sha,
                                    "gru_Wih_row_sha": wih_sha,
                                    "gru_Whh_row_sha": whh_sha,
                                    "expected_pack_row_sha": exp_pack_sha,
                                    "expected_Wih_row_sha": exp_wih_sha,
                                    "expected_Whh_row_sha": exp_whh_sha,
                                    "pack_row_mismatch": bool(pack_sha and exp_pack_sha and pack_sha != exp_pack_sha),
                                    "wih_row_mismatch": bool(wih_sha and exp_wih_sha and wih_sha != exp_wih_sha),
                                    "whh_row_mismatch": bool(whh_sha and exp_whh_sha and whh_sha != exp_whh_sha),
                                })
                            except Exception:
                                continue

                        self._eiel_emit_policy_bank_sync_probe(
                            tag=_tag, generation_index=_gen, tick=_tick,
                            kind="gru", stage="skip_not_dirty_details",
                            sync_info={
                                "n_gru": int(len(gru_agents)),
                                "token_mismatch_count": int(n_need_token),
                                "hmem_nonzero_count_sample": int(n_hmem_nonzero),
                                "gru_agent_ids_sha256_raw": _sha256_ints(gru_ids),
                                "gru_agent_ids_sha256_sorted": _sha256_ints_sorted(gru_ids),
                                "gru_slots_sha256_raw": _sha256_ints(gru_slots),
                                "gru_slots_sha256_sorted": _sha256_ints_sorted(gru_slots),
                                "gru_slot_tokens_sha256_raw": _sha256_ints(gru_tokens),
                                "gru_slot_tokens_sha256_sorted": _sha256_ints_sorted(gru_tokens),
                                "gru_bank_tokens_sha256_raw": _sha256_ints(gru_bank_tokens),
                                "gru_bank_tokens_sha256_sorted": _sha256_ints_sorted(gru_bank_tokens),
                                "gru_hmem_row_sha_sample_by_slot": hmem_row_sha,
                                "gru_pack_row_sha_sample_by_slot": gru_pack_row_sha,
                                "gru_Wih_row_sha_sample_by_slot": gru_wih_row_sha,
                                "gru_Whh_row_sha_sample_by_slot": gru_whh_row_sha,
                                "details_sample": det,
                            }
                        )
                except Exception:
                    pass


            # Determinism guard: token mismatch must force sync even if bank isn't marked dirty.
            # Rationale: slot_token!=bank_token implies slot reassignment or resume gap; skipping leaves stale rows.
            force_full = False
            try:
                for _a in gru_agents:
                    if bool(getattr(_a, '_eiel_force_full_gru_bank_sync_once', False)):
                        force_full = True
                        break
            except Exception:
                force_full = False
            n_need_token = 0
            mismatch_slots = []
            try:
                for _a in gru_agents:
                    try:
                        _slot = int(getattr(_a, 'slot', -1))
                        if _slot < 0:
                            continue
                        _tok = int(getattr(_a, 'slot_token', 0))
                        _last = int(getattr(_a, '_gru_bank_token', -1))
                        if _tok != _last:
                            n_need_token += 1
                            mismatch_slots.append(_slot)
                    except Exception:
                        continue
            except Exception:
                n_need_token = 0
            if (not force_full) and n_need_token == 0:
                if buckets is not None:
                    buckets["gru_bank_sync"] = buckets.get("gru_bank_sync", 0.0) + (time.perf_counter() - t0)
                return
            else:
                if _do_probe:
                    try:
                        self._eiel_emit_policy_bank_sync_probe(
                            tag=_tag, generation_index=_gen, tick=_tick,
                            kind='gru', stage='override_token_mismatch',
                            sync_info={'n_need_token': int(n_need_token), 'force_full': bool(force_full)}
                        )
                    except Exception:
                        pass
                # Fall through into normal sync path below (compute to_sync and apply).

        force_full = False
        for a in gru_agents:
            if bool(getattr(a, '_eiel_force_full_gru_bank_sync_once', False)):
                force_full = True
                break

        to_sync: List[object] = []
        for a in gru_agents:
            slot = int(getattr(a, "slot", -1))
            if slot < 0:
                continue
            if force_full:
                to_sync.append(a)
                continue
            token = int(getattr(a, "slot_token", 0))
            last = int(getattr(a, "_gru_bank_token", -1))
            if token != last:
                to_sync.append(a)

        if not to_sync:

            if _do_probe:
                try:
                    n_gru = int(len(gru_agents))
                    n_need = 0
                    for _a in gru_agents:
                        try:
                            tok = int(getattr(_a, "slot_token", 0))
                            last = int(getattr(_a, "_gru_bank_token", -1))
                            if tok != last:
                                n_need += 1
                        except Exception:
                            continue
                    self._eiel_emit_policy_bank_sync_probe(
                        tag=_tag, generation_index=_gen, tick=_tick,
                        kind="gru", stage="skip_no_to_sync",
                        sync_info={"n_gru": n_gru, "n_need": n_need, "cap_gru_pack": _cap0, "force_full": bool(force_full)}
                    )
                except Exception:
                    pass
            if buckets is not None:
                buckets["gru_bank_sync"] = buckets.get("gru_bank_sync", 0.0) + (time.perf_counter() - t0)
            return

        
        if _do_probe:
            try:
                ts_ids = [int(getattr(a, "agent_id", -1)) for a in to_sync]
                ts_slots = [int(getattr(a, "slot", -1)) for a in to_sync]
                ts_tokens = [int(getattr(a, "slot_token", 0)) for a in to_sync]
                self._eiel_emit_policy_bank_sync_probe(
                    tag=_tag, generation_index=_gen, tick=_tick,
                    kind="gru", stage="pre_apply",
                    sync_info={
                        "n_to_sync": int(len(to_sync)),
                        "to_sync_agent_ids_sha256": _sha256_ints(ts_ids),
                        "to_sync_agent_ids_sha256_sorted": _sha256_ints_sorted(ts_ids),
                        "to_sync_agent_ids_sha256_raw": _sha256_ints(ts_ids),
                        "to_sync_slots_sha256_sorted": _sha256_ints_sorted(ts_slots),
                        "to_sync_slots_sha256_raw": _sha256_ints(ts_slots),
                        "to_sync_slot_tokens_sha256_sorted": _sha256_ints_sorted(ts_tokens),
                        "to_sync_slot_tokens_sha256_raw": _sha256_ints(ts_tokens),
                        "to_sync_slots_sha256": _sha256_ints(ts_slots),
                        "to_sync_slot_tokens_sha256": _sha256_ints(ts_tokens),
                        "cap_gru_pack": _cap0,
                        "force_full": bool(force_full),
                        "skip_gru_memory_reset_once": bool(getattr(self, "_eiel_skip_gru_memory_reset_once", False)),
                    }
                )
            except Exception:
                pass
        if _do_probe and int(_tick) >= 100:
            _slot_agent_pairs_sha256_by_slot = ""
            try:
                det2 = []
                to_sync_set = set(int(getattr(a, "agent_id", -1)) for a in to_sync)
                for _a in gru_agents:
                    try:
                        aid = int(getattr(_a, "agent_id", -1))
                        det2.append({
                            "agent_id": aid,
                            "slot": int(getattr(_a, "slot", -1)),
                            "slot_token": int(getattr(_a, "slot_token", 0)),
                            "gru_bank_token": int(getattr(_a, "_gru_bank_token", -1)),
                            "in_to_sync": bool(aid in to_sync_set),
                        })
                    except Exception:
                        continue
                det2.sort(key=lambda d: (int(d.get("agent_id",-1)), int(d.get("slot",-1))))
                self._eiel_emit_policy_bank_sync_probe(
                    tag=_tag, generation_index=_gen, tick=_tick,
                    kind="gru", stage=f"tick{int(_tick):03d}_pre_apply_details",
                    sync_info={
                        "n_gru": int(len(gru_agents)),
                        "n_to_sync": int(len(to_sync)),
                        "details": det2,
                        "token_mismatch_count": int(sum(1 for d in det2 if int(d.get("slot_token", 0)) != int(d.get("gru_bank_token", -1)))),
                        "dupe_slots_in_gru_agents": _dupes([int(d.get("slot", -1)) for d in det2]),
                        "dupe_slots_in_to_sync": _dupes([int(getattr(a, "slot", -1)) for a in to_sync]),
                        "gru_bank_rows_sha_pre_by_slot": _gru_rows_sha([int(d.get("slot", -1)) for d in det2 if int(d.get("slot", -1)) >= 0]),
                        "hmem_by_slot_sha_pre": _hmem_slots_sha([int(d.get("slot",-1)) for d in det2 if int(d.get("slot",-1)) >= 0]),
                    }
                )
            except Exception:
                pass

        slots = torch.tensor([int(a.slot) for a in to_sync], device="cpu", dtype=torch.long)
        slots_gpu = slots.to(self._bank_device, non_blocking=True)

        encW = torch.stack([a.policy.encoder.weight.detach().cpu() for a in to_sync], dim=0)
        encb = torch.stack([a.policy.encoder.bias.detach().cpu() for a in to_sync], dim=0)

        Wih = torch.stack([a.policy.gru.weight_ih.detach().cpu() for a in to_sync], dim=0)
        Whh = torch.stack([a.policy.gru.weight_hh.detach().cpu() for a in to_sync], dim=0)
        bih = torch.stack([a.policy.gru.bias_ih.detach().cpu() for a in to_sync], dim=0)
        bhh = torch.stack([a.policy.gru.bias_hh.detach().cpu() for a in to_sync], dim=0)

        headW = torch.stack([a.policy.head.weight.detach().cpu() for a in to_sync], dim=0)
        headb = torch.stack([a.policy.head.bias.detach().cpu() for a in to_sync], dim=0)

        # Build packed parameter rows on CPU, then a single index_copy_ to GPU.
        pack_cpu = torch.cat(
            [
                encW.reshape(encW.shape[0], -1),
                encb.reshape(encb.shape[0], -1),
                Wih.reshape(Wih.shape[0], -1),
                bih.reshape(bih.shape[0], -1),
                Whh.reshape(Whh.shape[0], -1),
                bhh.reshape(bhh.shape[0], -1),
                headW.reshape(headW.shape[0], -1),
                headb.reshape(headb.shape[0], -1),
            ],
            dim=1,
        )

        # Single packed copy (main throughput win).
        if getattr(self, "_gru_pack", None) is not None:
            self._gru_pack.index_copy_(0, slots_gpu, pack_cpu.to(self._bank_device, non_blocking=True))

        self._encW.index_copy_(0, slots_gpu, encW.to(self._bank_device, non_blocking=True))
        self._encb.index_copy_(0, slots_gpu, encb.to(self._bank_device, non_blocking=True))
        self._Wih.index_copy_(0, slots_gpu, Wih.to(self._bank_device, non_blocking=True))
        self._Whh.index_copy_(0, slots_gpu, Whh.to(self._bank_device, non_blocking=True))
        self._bih.index_copy_(0, slots_gpu, bih.to(self._bank_device, non_blocking=True))
        self._bhh.index_copy_(0, slots_gpu, bhh.to(self._bank_device, non_blocking=True))
        self._headW.index_copy_(0, slots_gpu, headW.to(self._bank_device, non_blocking=True))
        self._headb.index_copy_(0, slots_gpu, headb.to(self._bank_device, non_blocking=True))

        # For new/changed agents, reset their memory state.
        #
        # Resume determinism: allow a one-shot skip so restored GRU memory (_Hmem)
        # is not zeroed during the first post-resume bank sync.
        _did_skip_reset = False
        if bool(getattr(self, "_eiel_skip_gru_memory_reset_once", False)):
            _did_skip_reset = True
            try:
                setattr(self, "_eiel_skip_gru_memory_reset_once", False)
            except Exception:
                pass
        else:
            self._reset_memory_slots(slots_gpu)

        for a in to_sync:
            a._gru_bank_token = int(a.slot_token)
            if force_full:
                try:
                    a._eiel_force_full_gru_bank_sync_once = False
                except Exception:
                    pass

        self._bank_dirty_gru = False

        if _do_probe and int(_tick) >= 100:
            _slot_agent_pairs_sha256_by_slot = ""
            try:
                det3 = []
                to_sync_set = set(int(getattr(a, "agent_id", -1)) for a in to_sync)
                for _a in gru_agents:
                    try:
                        aid = int(getattr(_a, "agent_id", -1))
                        det3.append({
                            "agent_id": aid,
                            "slot": int(getattr(_a, "slot", -1)),
                            "slot_token": int(getattr(_a, "slot_token", 0)),
                            "gru_bank_token": int(getattr(_a, "_gru_bank_token", -1)),
                            "in_to_sync": bool(aid in to_sync_set),
                        })
                    except Exception:
                        continue
                det3.sort(key=lambda d: (int(d.get("agent_id",-1)), int(d.get("slot",-1))))
                self._eiel_emit_policy_bank_sync_probe(
                    tag=_tag, generation_index=_gen, tick=_tick,
                    kind="gru", stage=f"tick{int(_tick):03d}_post_apply_details",
                    sync_info={
                        "n_gru": int(len(gru_agents)),
                        "n_to_sync": int(len(to_sync)),
                        "details": det3,
                        "token_mismatch_count": int(sum(1 for d in det3 if int(d.get("slot_token", 0)) != int(d.get("gru_bank_token", -1)))),
                        "dupe_slots_in_gru_agents": _dupes([int(d.get("slot", -1)) for d in det3]),
                        "dupe_slots_in_to_sync": _dupes([int(getattr(a, "slot", -1)) for a in to_sync]),
                        "gru_bank_rows_sha_post_by_slot": _gru_rows_sha([int(d.get("slot", -1)) for d in det3 if int(d.get("slot", -1)) >= 0]),
                        "hmem_by_slot_sha_post": _hmem_slots_sha([int(d.get("slot",-1)) for d in det3 if int(d.get("slot",-1)) >= 0]),
                        "did_skip_memory_reset": bool(_did_skip_reset),
                    }
                )
            except Exception:
                pass

        if _do_probe:
            try:
                cap1 = int(getattr(self._gru_pack, 'shape', [0])[0]) if getattr(self, '_gru_pack', None) is not None else 0
                self._eiel_emit_policy_bank_sync_probe(
                    tag=_tag, generation_index=_gen, tick=_tick,
                    kind="gru", stage="post_apply",
                    sync_info={
                        "cap_gru_pack": cap1,
                        "n_to_sync": int(len(to_sync)),
                        "force_full": bool(force_full),
                        "skip_gru_memory_reset_once": bool(getattr(self, "_eiel_skip_gru_memory_reset_once", False)),
                    }
                )
            except Exception:
                pass

        if buckets is not None:
            buckets["gru_bank_sync"] = buckets.get("gru_bank_sync", 0.0) + (time.perf_counter() - t0)
