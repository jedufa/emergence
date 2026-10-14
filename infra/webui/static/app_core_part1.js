window.__EIEL_APP_CORE_PART1_LOADED__ = true;
window.__EIEL_APP_CORE_JS_LOADED__ = true;

/* infra/webui/static/app_core.js
 * EIEL Phase 5d Tier-1 Web UI
 *
 * Frontend goals:
 * - boring, robust, vanilla JS
 * - read-only on determinism-critical artifacts
 * - all writes are UI-only (/ui) or inbox submissions (/inbox) via server
 *
 * NOTE: This file must never crash on missing DOM nodes.
 * (Because browsers are great at failing in creative ways.)
 *
 * Last modified: 2026-03-05 Europe/Vienna
 */

"use strict";

// Boot marker: used by index.html to detect broken /static/app_core.js loads.
// If /static/app_core.js 404s (or is cached incorrectly), the UI would otherwise look "empty".
// Marker moved to the first line (before any parsing/execution surprises).
// Keep legacy marker assignment for compatibility.
try{ window.__EIEL_APP_CORE_JS_LOADED__ = true; }catch(_e){}

// -----------------------------
// Shared helpers
// -----------------------------

const __shared = (window.EIELShared && typeof window.EIELShared === "object") ? window.EIELShared : {};
const q = __shared.q || function(sel, root){ return (root || document).querySelector(sel); };
const qa = __shared.qa || function(sel, root){ return Array.from((root || document).querySelectorAll(sel)); };
const safeOn = __shared.safeOn || function(el, evt, fn){ if(el) el.addEventListener(evt, fn); };
const escapeHtml = __shared.escapeHtml || function(s){ return String(s).replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c])); };
const badgeForStatus = __shared.badgeForStatus || function(status){ return `<span class="badge badge-muted">${escapeHtml(String(status || ""))}</span>`; };
const badgeForDeterminism = __shared.badgeForDeterminism || function(det){ return `<span class="badge badge-muted">${escapeHtml(String(det || ""))}</span>`; };
const badgeForStrict = __shared.badgeForStrict || function(strict){ return strict ? '<span class="badge badge-accent">STRICT</span>' : '<span class="badge badge-muted">NON-STRICT</span>'; };
const showGlobalError = __shared.showGlobalError || function(title, msg){ try{ console.error(String(title || "Error"), String(msg || "")); }catch(_e){} };
const clearGlobalError = __shared.clearGlobalError || function(){};
const showToast = __shared.showToast || function(title, body){ try{ console.log(String(title || ""), String(body || "")); }catch(_e){} };
const toast = __shared.toast || function(title, body, ms){ showToast(title, body, ms); };
const setHealth = __shared.setHealth || function(_ok){};
const openModal = __shared.openModal || function(_title, _bodyHtml, _onOpen){};
const closeModal = __shared.closeModal || function(){};
const fetchText = __shared.fetchText || (async function(url, opts){ const r = await fetch(url, opts); return await r.text(); });
const fetchJson = __shared.fetchJson || (async function(url, opts){ return JSON.parse(await fetchText(url, opts)); });
const postJson = __shared.postJson || (async function(url, obj){ return await fetchJson(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(obj || {})}); });

const __utils = (window.EIELUtils && typeof window.EIELUtils === "object") ? window.EIELUtils : {};
const copyTextToClipboard = __utils.copyTextToClipboard || (async function(txt){
  const t = String(txt ?? "");
  try{
    if(globalThis.navigator && navigator.clipboard && typeof navigator.clipboard.writeText === "function"){
      await navigator.clipboard.writeText(t);
      return true;
    }
  }catch(_e){}
  try{
    const ta = document.createElement("textarea");
    ta.value = t;
    ta.setAttribute("readonly", "readonly");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    ta.style.top = "0";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
    return true;
  }catch(_e){}
  return false;
});
const shortId = __utils.shortId || function(s, max){ const t = String(s || ""); const m = max || 26; return t.length <= m ? t : (t.slice(0, m - 1) + "…"); };
const displayId = __utils.displayId || function(obj, opts){ const fallback = String(opts?.fallback || ""); return String(obj?.display_id || fallback || obj?.run_key || obj?.job_id || obj?.name || "item"); };
const fmtNum = __utils.fmtNum || function(x, digits){ if(x === null || x === undefined || x === "") return ""; const n = Number(x); if(!Number.isFinite(n)) return ""; const d = (digits === undefined) ? 3 : digits; return n.toFixed(d).replace(/\.0+$/,"").replace(/(\.[0-9]*?)0+$/,"$1"); };
const fmtGen = __utils.fmtGen || function(x){ const n = parseInt(String(x ?? ""), 10); return (Number.isFinite(n) && n >= 0) ? String(n) : ""; };
const csvListToArray = __utils.csvListToArray || function(s){ const raw = String(s || "").trim(); return raw ? raw.split(",").map(x=>x.trim()).filter(Boolean) : []; };
const maybeInt = __utils.maybeInt || function(s){ const t = String(s ?? "").trim(); if(!t) return null; const n = parseInt(t, 10); return Number.isFinite(n) ? n : null; };
const genUuid = __utils.genUuid || function(){ try{ if(globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") return globalThis.crypto.randomUUID(); }catch(_e){} const r = () => Math.floor(Math.random()*0xffffffff).toString(16).padStart(8,"0"); return `${r()}-${r().slice(0,4)}-${r().slice(0,4)}-${r().slice(0,4)}-${r()}${r().slice(0,4)}`; };
const badge = __utils.badge ? ((text, cls, ttKey)=>__utils.badge(text, cls, ttKey, escapeHtml)) : function(text, cls, ttKey){ const tt = String(ttKey||"").trim(); const ttAttr = tt ? ` data-tt="${escapeHtml(tt)}" tabindex="0"` : ""; return `<span class="badge ${cls || ""}"${ttAttr}>${escapeHtml(text)}</span>`; };
const chip = __utils.chip ? ((text, ttKey)=>__utils.chip(text, ttKey, escapeHtml)) : function(text, ttKey){ const tt = String(ttKey||"").trim(); const ttAttr = tt ? ` data-tt="${escapeHtml(tt)}" tabindex="0"` : ""; return `<span class="chip"${ttAttr}>${escapeHtml(text)}</span>`; };
const statusBadgeClass = __utils.statusBadgeClass || function(status){ const s = String(status || "").toUpperCase(); if(s === "RUNNING") return "b-running"; if(s === "PLANNED") return "b-planned"; if(s === "COMPLETE") return "b-complete"; if(s === "FAILED") return "b-failed"; if(s === "STALLED") return "b-stalled"; if(s === "INCOMPLETE") return "b-incomplete"; if(s === "POSSIBLY_CORRUPT") return "b-corrupt"; return "b-unknown"; };
const downloadTextFile = __utils.downloadTextFile ? ((filename, text)=>__utils.downloadTextFile(filename, text, showToast)) : function(filename, text){ try{ const blob = new Blob([String(text||"")], {type:"text/plain;charset=utf-8"}); const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = String(filename||"download.txt"); document.body.appendChild(a); a.click(); a.remove(); setTimeout(()=>{ try{ URL.revokeObjectURL(url); }catch(_e){} }, 5000);}catch(e){ showToast("Download failed", String(e?.message||e)); } };
const __api = (window.EIELApiFactory && typeof window.EIELApiFactory.create === "function")
  ? window.EIELApiFactory.create({
      fetchJson: fetchJson,
      postJson: postJson,
      encRunKey: function(v){ return encodeURIComponent(String(v || "")); },
    })
  : {};
function _missingApi(name){
  return async function(){
    throw new Error("Missing API wrapper: " + String(name || "unknown"));
  };
}

// -----------------------------
// Run Builder (Phase 5e Tier-1)
// -----------------------------

const Builder = {
  loaded: false,
  domReady: false,
  cfg: null,
  registries: [],
  templates: [],
  knobKeys: [],
  // In-memory harness axis flags (axis -> value -> tokens[])
  axisFlags: {},
  _persistTimer: null,
  launchInflight: false,
  launchLifecycle: "IDLE",
  activeLaunchJobId: "",
  launchLastStatus: null,
  launchPollTimer: null,
  launchPollCount: 0,
  launchPollErrorCount: 0,
  launchAttemptSeq: 0,
  launchRequestId: "",
  launchTraceStep: "",
  startButtonsForceDisabled: false,
  startBlockedReason: "",
  // True only when the harness start command came from a successful
  // /api/builder/generate_harness response for the current form state.
  harnessLaunchReady: false,
  selectedRegistryPath: "",
  loadedRegistryPath: "",
  loadedHarnessPath: "",
  loadedHarnessExists: false,
  harnessBaselineSignature: "",
  harnessDirty: false,
};

async function apiBuilderConfig(){
  return await ((__api.apiBuilderConfig || _missingApi("apiBuilderConfig"))());
}
async function apiBuilderRegistries(){
  return await ((__api.apiBuilderRegistries || _missingApi("apiBuilderRegistries"))());
}
async function apiBuilderTemplates(){
  return await ((__api.apiBuilderTemplates || _missingApi("apiBuilderTemplates"))());
}
async function apiBuilderSaveTemplate(obj){
  return await ((__api.apiBuilderSaveTemplate || _missingApi("apiBuilderSaveTemplate"))(obj));
}
async function apiBuilderLoadTemplate(obj){
  return await ((__api.apiBuilderLoadTemplate || _missingApi("apiBuilderLoadTemplate"))(obj));
}
async function apiBuilderGenerateHarness(obj){
  return await ((__api.apiBuilderGenerateHarness || _missingApi("apiBuilderGenerateHarness"))(obj));
}
async function apiBuilderGenerateManifest(obj){
  return await ((__api.apiBuilderGenerateManifest || _missingApi("apiBuilderGenerateManifest"))(obj));
}
async function apiBuilderRegistryLoad(obj){
  return await ((__api.apiBuilderRegistryLoad || _missingApi("apiBuilderRegistryLoad"))(obj));
}
async function apiBuilderRegistryDeletePreview(obj){
  return await ((__api.apiBuilderRegistryDeletePreview || _missingApi("apiBuilderRegistryDeletePreview"))(obj));
}
async function apiBuilderRegistryDelete(obj){
  return await ((__api.apiBuilderRegistryDelete || _missingApi("apiBuilderRegistryDelete"))(obj));
}
async function apiBuilderRegistryParse(obj){
  return await ((__api.apiBuilderRegistryParse || _missingApi("apiBuilderRegistryParse"))(obj));
}

async function apiBuilderLaunch(obj){
  return await ((__api.apiBuilderLaunch || _missingApi("apiBuilderLaunch"))(obj));
}
async function apiBuilderHarnesses(){
  return await ((__api.apiBuilderHarnesses || _missingApi("apiBuilderHarnesses"))());
}
async function apiBuilderJobStatus(jobId){
  return await ((__api.apiBuilderJobStatus || _missingApi("apiBuilderJobStatus"))(jobId));
}
async function apiBuilderJobLog(jobId, maxBytes){
  return await ((__api.apiBuilderJobLog || _missingApi("apiBuilderJobLog"))(jobId, maxBytes));
}
async function apiSystemProcesses(includeAll){
  return await ((__api.apiSystemProcesses || _missingApi("apiSystemProcesses"))(includeAll));
}
async function apiRunFileTail(runKey, which, maxBytes){
  return await ((__api.apiRunFileTail || _missingApi("apiRunFileTail"))(runKey, which, maxBytes));
}
async function apiSystemTerminateProcess(pid){
  return await ((__api.apiSystemTerminateProcess || _missingApi("apiSystemTerminateProcess"))(pid));
}
async function apiSystemKillProcess(pid){
  return await ((__api.apiSystemKillProcess || _missingApi("apiSystemKillProcess"))(pid));
}

function _builderExtractJsonErrorPayload(err){
  const msg = String(err?.message || err || "").trim();
  if(!msg) return null;
  const marker = " :: ";
  const idx = msg.indexOf(marker);
  if(idx < 0) return null;
  const raw = msg.slice(idx + marker.length).trim();
  if(!raw || raw[0] !== "{") return null;
  try{
    const parsed = JSON.parse(raw);
    return (parsed && typeof parsed === "object") ? parsed : null;
  }catch(_e){
    return null;
  }
}

function _builderLaunchErrorMessage(err){
  const payload = _builderExtractJsonErrorPayload(err);
  if(payload && String(payload.error || "") === "launch_preflight_failed"){
    const reason = String(payload.message || "").trim() || "Launch preflight failed.";
    const fc = String(payload.failure_code || "").trim();
    const mod = String(payload.module || "").trim();
    const mp = String(payload?.details?.manifest_path || payload.launch_manifest_path || "").trim();
    const lines = [reason];
    if(fc) lines.push(`Failure code: ${fc}`);
    if(mod) lines.push(`Module: ${mod}`);
    if(mp) lines.push(`Manifest: ${mp}`);
    return lines.join("\n");
  }
  if(payload && String(payload.error || "") === "launch_failed_startup"){
    const reason = String(payload.message || "").trim() || "Process exited during startup.";
    const exitCode = Number(payload.exit_code);
    const logTail = String(payload.log_tail || "").trim();
    const logPath = String(payload.log_path || "").trim();
    const lines = [reason];
    if(Number.isFinite(exitCode)) lines.push(`Exit code: ${exitCode}`);
    if(logTail) lines.push(`Log tail: ${logTail.slice(0, 500)}`);
    if(logPath) lines.push(`Log path: ${logPath}`);
    return lines.join("\n");
  }
  return String(err?.message || err || "Launch failed");
}

function _builderSetLaunchTrace(step, requestId){
  Builder.launchTraceStep = String(step || "").trim();
  if(typeof requestId !== "undefined" && requestId !== null){
    Builder.launchRequestId = String(requestId || "").trim();
  }
  try{
    if(Builder.launchTraceStep || Builder.launchRequestId){
      console.debug("[builder-launch-trace]", {
        step: String(Builder.launchTraceStep || ""),
        request_id: String(Builder.launchRequestId || ""),
        lifecycle: String(Builder.launchLifecycle || ""),
        job_id: String(Builder.activeLaunchJobId || ""),
      });
    }
  }catch(_e){}
}

function _builderStartButtons(){
  return [
    q("#rbCmdStart"),
    q("#rbHarnessCmdStart"),
    q("#rbManifestCmdStart"),
  ].filter(Boolean);
}

function _builderSetStartButtonsBusy(busy, sourceBtn){
  const on = !!busy;
  for(const btn of _builderStartButtons()){
    if(!btn) continue;
    const ds = (btn.dataset && typeof btn.dataset === "object") ? btn.dataset : (btn.dataset = {});
    const cls = (btn.classList && typeof btn.classList === "object") ? btn.classList : null;
    const current = String(btn.textContent || "").trim();
    if(on){
      if(!ds._rbOrigLabel){
        ds._rbOrigLabel = current || "Start";
      }
      try{ if(cls && typeof cls.add === "function") cls.add("is-busy"); }catch(_e){}
      btn.textContent = (sourceBtn && btn === sourceBtn) ? "Starting..." : "Locked...";
      try{
        if(typeof btn.setAttribute === "function"){
          btn.setAttribute("aria-busy", "true");
        }else{
          btn["aria-busy"] = "true";
        }
      }catch(_e){}
    }else{
      const orig = String(ds._rbOrigLabel || "").trim();
      if(orig){
        btn.textContent = orig;
      }
      try{ if(cls && typeof cls.remove === "function") cls.remove("is-busy"); }catch(_e){}
      try{
        if(typeof btn.removeAttribute === "function"){
          btn.removeAttribute("aria-busy");
        }else{
          delete btn["aria-busy"];
        }
      }catch(_e){}
      try{ delete ds._rbOrigLabel; }catch(_e){ ds._rbOrigLabel = ""; }
    }
  }
}

function _builderReadTextLike(el){
  if(!el) return "";
  if(typeof el.value === "string") return String(el.value || "");
  return String(el.textContent || "");
}

function _builderWriteTextLike(el, text){
  if(!el) return;
  const next = String(text || "");
  if(typeof el.value === "string"){
    el.value = next;
  }else{
    el.textContent = next;
    try{ el.title = next; }catch(_e){}
  }
}

function _builderShellSplit(cmd){
  const src = String(cmd || "");
  const out = [];
  let cur = "";
  let quote = "";
  for(let i = 0; i < src.length; i += 1){
    const ch = src[i];
    if(quote){
      cur += ch;
      if(ch === "\\" && i + 1 < src.length){
        i += 1;
        cur += src[i];
        continue;
      }
      if(ch === quote) quote = "";
      continue;
    }
    if(ch === "'" || ch === '"'){
      quote = ch;
      cur += ch;
      continue;
    }
    if(/\s/.test(ch)){
      if(cur){
        out.push(cur);
        cur = "";
      }
      continue;
    }
    if(ch === "\\" && i + 1 < src.length){
      cur += ch;
      i += 1;
      cur += src[i];
      continue;
    }
    cur += ch;
  }
  if(cur) out.push(cur);
  return out;
}

function _builderFormatCommandForDisplay(text){
  const src = String(text || "").trim();
  if(!src) return "";
  const tokens = _builderShellSplit(src);
  if(!tokens.length) return src;
  const groups = [];
  let i = 0;
  const envs = [];
  while(i < tokens.length && /^[A-Za-z_][A-Za-z0-9_]*=/.test(tokens[i])){
    envs.push(tokens[i]);
    i += 1;
  }
  if(envs.length){
    groups.push(envs.join(" "));
  }
  if(i < tokens.length){
    if(tokens[i] === "python" && tokens[i + 1] === "-m" && tokens[i + 2]){
      groups.push(tokens.slice(i, i + 3).join(" "));
      i += 3;
    }else{
      groups.push(tokens[i]);
      i += 1;
    }
  }
  while(i < tokens.length){
    const tok = tokens[i];
    if(/^--[^=]+=/.test(tok) || /^-[^-]=/.test(tok)){
      groups.push(tok);
      i += 1;
      continue;
    }
    if(tok.startsWith("-")){
      if(i + 1 < tokens.length && !tokens[i + 1].startsWith("-")){
        groups.push(`${tok} ${tokens[i + 1]}`);
        i += 2;
      }else{
        groups.push(tok);
        i += 1;
      }
      continue;
    }
    groups.push(tok);
    i += 1;
  }
  return groups.join("\n");
}

function _builderWriteCommandPreview(el, rawText){
  if(!el) return;
  const shown = _builderFormatCommandForDisplay(rawText);
  _builderWriteTextLike(el, shown);
  if(typeof el.value === "string"){
    try{
      const rows = Math.max(3, Math.min(8, shown ? shown.split(/\n/).length : 2));
      el.rows = rows;
    }catch(_e){}
  }
}

function _builderPreviewOrFallback(selector, rawFallback){
  const shown = String(_builderReadTextLike(q(selector)) || "").trim();
  if(shown) return shown;
  const raw = String(rawFallback || "").trim();
  return raw ? _builderFormatCommandForDisplay(_builderApplyEnvPrefix(_builderCmdToManaged(raw))) : "";
}

function _builderReadinessState(){
  const cmdPreview = _builderPreviewOrFallback("#rbCmdOut", Builder.lastCliCmdRaw || "");
  const harnessPreview = _builderPreviewOrFallback("#rbHarnessCmdOut", Builder.lastHarnessCmdRaw || "");
  const manifestPreview = _builderPreviewOrFallback("#rbManifestCmdOut", Builder.lastManifestCmdRaw || "");

  const cmd = {
    ready: !!String(cmdPreview || "").trim(),
    reason: "Generate a command first.",
  };
  const harness = {
    ready: (!!String(harnessPreview || "").trim()) && !!Builder.harnessLaunchReady && !Builder.harnessDirty,
    reason: "Generate harness or load one from the list, then Start (command is stale after edits).",
  };
  const manifest = {
    ready: !!String(manifestPreview || "").trim(),
    reason: "Generate a manifest first.",
  };
  const blocked = [cmd, harness, manifest].find((x)=>!x.ready);
  return {
    cmd,
    harness,
    manifest,
    blocked_reason: blocked ? String(blocked.reason || "").trim() : "",
  };
}

function _builderApplyStartButtonState(){
  const bySel = {
    "#rbCmdStart": "cmd",
    "#rbHarnessCmdStart": "harness",
    "#rbManifestCmdStart": "manifest",
  };
  const readiness = _builderReadinessState();
  const activeJid = String(Builder.activeLaunchJobId || "").trim();
  const startingLike = (Builder.launchLifecycle === "STARTING");
  // Guard against stale in-memory lifecycle that can survive transient UI paths.
  if((startingLike || Builder.launchLifecycle === "RUNNING") && !activeJid && !Builder.launchInflight){
    Builder.launchLifecycle = "IDLE";
  }
  // Allow launching a new run even if an older managed job is still RUNNING.
  // Hard block only while a launch request is actively in flight/starting.
  const hardBlock = !!(Builder.startButtonsForceDisabled || Builder.launchInflight || (startingLike && !!activeJid));
  const hardReason = "A launch is already in flight.";
  let softReason = "";
  for(const sel of Object.keys(bySel)){
    const btn = q(sel);
    if(!btn) continue;
    const key = bySel[sel];
    const readyObj = readiness[key] || {ready: true, reason: ""};
    const disabled = hardBlock || !readyObj.ready;
    btn.disabled = !!disabled;
    const reason = hardBlock ? hardReason : (readyObj.ready ? "" : String(readyObj.reason || "").trim());
    try{
      btn.title = reason ? `Start blocked: ${reason}` : "Start";
    }catch(_e){}
    if((!hardBlock) && (!readyObj.ready) && (!softReason)){
      softReason = String(readyObj.reason || "").trim();
    }
  }
  Builder.startBlockedReason = hardBlock ? hardReason : softReason;
}

function _builderSetStartButtonsDisabled(disabled){
  Builder.startButtonsForceDisabled = !!disabled;
  _builderApplyStartButtonState();
}

function _builderStatusFailureDetails(st){
  if(!st || typeof st !== "object") return "";
  const lines = [];
  const fc = String(st.failure_code || "").trim();
  const fd = String(st.failure_detail || "").trim();
  const ec = Number(st.exit_code);
  const lp = String(st.log_path || "").trim();
  const lt = String(st.log_tail || "").trim();
  if(Number.isFinite(ec)) lines.push(`Exit code: ${ec}`);
  if(fc) lines.push(`Failure code: ${fc}`);
  if(fd) lines.push(`Failure detail: ${fd}`);
  if(lt) lines.push(`Log tail: ${lt.slice(0, 500)}`);
  if(lp) lines.push(`Log path: ${lp}`);
  return lines.join("\n");
}

function _builderRenderLaunchStatus(){
  const box = q("#rbLaunchStatus");
  if(!box) return;
  _builderApplyStartButtonState();
  const lc = String(Builder.launchLifecycle || "IDLE").toUpperCase();
  const jid = String(Builder.activeLaunchJobId || "").trim();
  const parts = [];
  if(jid) parts.push(`job=${jid}`);
  parts.push(`state=${lc}`);
  const st = Builder.launchLastStatus || {};
  const ec = Number(st.exit_code);
  if(Number.isFinite(ec)) parts.push(`exit=${ec}`);
  const fc = String(st.failure_code || "").trim();
  if(fc) parts.push(`failure=${fc}`);
  const rid = String(Builder.launchRequestId || st.request_id || "").trim();
  if(rid) parts.push(`req=${rid}`);
  const step = String(Builder.launchTraceStep || "").trim();
  if(step) parts.push(`step=${step}`);
  const blocked = String(Builder.startBlockedReason || "").trim();
  if(blocked && lc !== "STARTING" && lc !== "RUNNING"){
    parts.push(`blocked=${blocked}`);
  }
  box.textContent = parts.join(" | ");
}

function _builderClearLaunchPoll(){
  if(Builder.launchPollTimer){
    try{ clearTimeout(Builder.launchPollTimer); }catch(_e){}
    Builder.launchPollTimer = null;
  }
}

function _builderApplyLaunchStatus(st){
  const status = (st && typeof st === "object") ? st : {};
  const lc = String(status.lifecycle || "").trim().toUpperCase();
  if(lc) Builder.launchLifecycle = lc;
  if(String(status.request_id || "").trim()){
    Builder.launchRequestId = String(status.request_id || "").trim();
  }
  _builderSetLaunchTrace(`poll_${String(Builder.launchLifecycle || "unknown").toLowerCase()}`);
  Builder.launchLastStatus = status;
  _builderRenderLaunchStatus();

  const terminal = !!status.terminal;
  _builderApplyStartButtonState();

  if(!terminal) return;

  _builderClearLaunchPoll();
  const wasJob = String(Builder.activeLaunchJobId || "").trim();
  Builder.activeLaunchJobId = "";
  Builder.launchPollCount = 0;
  Builder.launchPollErrorCount = 0;
  _builderPersistSoon();
  _builderSetStartButtonsDisabled(false);

  if(Builder.launchLifecycle === "FAILED"){
    const details = _builderStatusFailureDetails(status);
    if(details) showGlobalError("Run failed", details);
    else showGlobalError("Run failed", "Managed launch reached FAILED state.");
  }else if(Builder.launchLifecycle === "SUCCEEDED"){
    showToast("Run complete", wasJob ? `job ${wasJob}` : "managed job finished");
  }
  refreshRuns();
}

function _builderScheduleLaunchPoll(immediate){
  _builderClearLaunchPoll();
  const jid = String(Builder.activeLaunchJobId || "").trim();
  if(!jid) return;
  _builderSetLaunchTrace(immediate ? "poll_schedule_immediate" : "poll_schedule");
  if(Builder.launchPollCount >= 900){
    _builderSetStartButtonsDisabled(false);
    return;
  }
  const delay = immediate ? 120 : 1200;
  Builder.launchPollTimer = setTimeout(async ()=>{
    Builder.launchPollTimer = null;
    const activeJid = String(Builder.activeLaunchJobId || "").trim();
    if(!activeJid) return;
    Builder.launchPollCount += 1;
    try{
      _builderSetLaunchTrace("poll_request");
      const st = await apiBuilderJobStatus(activeJid);
      Builder.launchPollErrorCount = 0;
      _builderApplyLaunchStatus(st);
      if(!st?.terminal){
        _builderScheduleLaunchPoll(false);
      }
    }catch(e){
      _builderSetLaunchTrace("poll_error");
      const msg = String(e?.message || e || "");
      if(msg.toLowerCase().includes("404")){
        Builder.launchLifecycle = "STOPPED";
        Builder.launchLastStatus = null;
        Builder.activeLaunchJobId = "";
        Builder.launchPollErrorCount = 0;
        _builderRenderLaunchStatus();
        _builderSetStartButtonsDisabled(false);
        _builderClearLaunchPoll();
        _builderPersistSoon();
        return;
      }
      Builder.launchPollErrorCount = (Number(Builder.launchPollErrorCount) || 0) + 1;
      if(Builder.launchPollErrorCount >= 5){
        Builder.launchLifecycle = "STOPPED";
        Builder.launchLastStatus = {
          lifecycle: "STOPPED",
          terminal: true,
          failure_code: "JOB_STATUS_POLL_FAILED",
          failure_detail: msg.slice(0, 500),
        };
        Builder.activeLaunchJobId = "";
        _builderRenderLaunchStatus();
        _builderSetStartButtonsDisabled(false);
        _builderClearLaunchPoll();
        _builderPersistSoon();
        showGlobalError("Launch status polling failed", msg.slice(0, 500));
        return;
      }
      _builderScheduleLaunchPoll(false);
    }
  }, delay);
}

async function _builderStartManagedCommand(cmd, label, sourceBtn){
  const c = String(cmd || "").trim();
  if(!c){
    showToast("Nothing to start", "Generate a command first");
    return;
  }
  if(Builder.launchInflight || Builder.launchLifecycle === "STARTING"){
    const activeJid = String(Builder.activeLaunchJobId || "").trim();
    if(activeJid){
      _builderSetLaunchTrace("start_lock_recheck");
      try{
        const st = await apiBuilderJobStatus(activeJid);
        _builderApplyLaunchStatus(st);
      }catch(_e){
        // If status recheck fails, clear stale lock and allow retry.
        Builder.activeLaunchJobId = "";
        Builder.launchLifecycle = "IDLE";
        Builder.launchLastStatus = null;
        Builder.launchInflight = false;
        _builderSetStartButtonsDisabled(false);
        _builderRenderLaunchStatus();
      }
    }
    if(!(Builder.launchInflight || Builder.launchLifecycle === "STARTING")){
      _builderSetLaunchTrace("start_lock_cleared_retry");
      return await _builderStartManagedCommand(c, label);
    }
    _builderSetLaunchTrace("start_blocked_lock");
    showToast("Launch locked", "A launch is already in flight or running.");
    return;
  }
  Builder.launchInflight = true;
  Builder.launchAttemptSeq = (Number(Builder.launchAttemptSeq) || 0) + 1;
  const attemptSeq = Number(Builder.launchAttemptSeq) || 1;
  Builder.launchRequestId = "";
  _builderSetLaunchTrace("start_click");
  Builder.launchLifecycle = "STARTING";
  Builder.launchLastStatus = null;
  try{
    _builderSetStartButtonsBusy(true, sourceBtn);
    _builderSetStartButtonsDisabled(true);
    _builderRenderLaunchStatus();
  }catch(_e){}
  let finalized = false;
  try{
    const managed = true;
    _builderSetLaunchTrace("launch_request_sent");
    const launchTimeoutMs = 15000;
    const timeoutPromise = new Promise((_, reject)=>{
      setTimeout(()=>reject(new Error(`launch request timed out after ${launchTimeoutMs}ms`)), launchTimeoutMs);
    });
    const res = await Promise.race([apiBuilderLaunch({cmd: c, managed}), timeoutPromise]);
    if((Number(Builder.launchAttemptSeq) || 0) !== attemptSeq){
      return;
    }
    const rid = String(res?.request_id || "").trim();
    _builderSetLaunchTrace("launch_response_ok", rid);
    const pid = res && res.pid ? `pid ${res.pid}` : "started";
    const jid = String(res?.job_id || "").trim();
    Builder.launchInflight = false;
    Builder.launchPollCount = 0;
    Builder.launchPollErrorCount = 0;
    Builder.activeLaunchJobId = jid;
    Builder.launchLifecycle = jid ? "STARTING" : "RUNNING";
    _builderPersistSoon();
    _builderRenderLaunchStatus();
    if(jid){
      _builderSetLaunchTrace("launch_poll_begin", rid);
      _builderScheduleLaunchPoll(true);
      showToast(`${label} launched`, `job ${jid} (${pid})`);
    }else{
      _builderSetLaunchTrace("launch_started_no_job", rid);
      _builderSetStartButtonsDisabled(false);
      showToast(`${label} launched`, pid);
    }
    refreshRuns();
    finalized = true;
  }catch(e){
    if((Number(Builder.launchAttemptSeq) || 0) !== attemptSeq){
      return;
    }
    const payload = _builderExtractJsonErrorPayload(e);
    const rid = String(payload?.request_id || "").trim();
    _builderSetLaunchTrace("launch_response_error", rid);
    Builder.launchInflight = false;
    Builder.launchLifecycle = "FAILED";
    Builder.launchLastStatus = payload || null;
    _builderSetStartButtonsDisabled(false);
    _builderRenderLaunchStatus();
    showGlobalError("Launch failed", _builderLaunchErrorMessage(e));
    finalized = true;
  }finally{
    // Safety net: never leave Start buttons locked forever on unexpected paths.
    if((Number(Builder.launchAttemptSeq) || 0) === attemptSeq){
      if(Builder.launchInflight && (!finalized || !String(Builder.activeLaunchJobId || "").trim())){
        Builder.launchInflight = false;
        if(String(Builder.launchLifecycle || "").toUpperCase() === "STARTING" && !String(Builder.activeLaunchJobId || "").trim()){
          Builder.launchLifecycle = "IDLE";
        }
        _builderSetLaunchTrace("launch_safety_unlock");
        _builderSetStartButtonsDisabled(false);
        _builderRenderLaunchStatus();
      }
      _builderSetStartButtonsBusy(false, null);
    }
  }
}

function _parseIntStrict(val, fallback){
  const n = parseInt(String(val ?? "").trim(), 10);
  return Number.isFinite(n) ? n : fallback;
}

function _parseFloatStrict(val, fallback){
  const n = Number.parseFloat(String(val ?? "").trim());
  return Number.isFinite(n) ? n : fallback;
}

function _parseKnobValue(raw){
  const s = String(raw ?? "").trim();
  if(!s) return s;
  const lo = s.toLowerCase();
  if(lo === "true") return true;
  if(lo === "false") return false;
  const n = Number(s);
  if(Number.isFinite(n)) return n;
  return s;
}

function derivePressureAutopilotActiveState(){
  const keys = [
    "pressure_autopilot_enable",
    "pressure_autopilot_tick_interval_gen",
    "pressure_autopilot_profile",
    "pressure_autopilot_delta_up",
    "pressure_autopilot_delta_down",
    "pressure_autopilot_target_survival",
    "pressure_autopilot_survival_ceiling",
  ];
  const pathToKey = new Map(keys.map(k => [`sim_cfg.${k}`, k]));

  const defaults = {
    pressure_autopilot_enable: false,
    pressure_autopilot_tick_interval_gen: 500,
    pressure_autopilot_profile: "balanced",
    pressure_autopilot_delta_up: 0.05,
    pressure_autopilot_delta_down: 0.10,
    pressure_autopilot_target_survival: 0.55,
    pressure_autopilot_survival_ceiling: 0.80,
  };

  const values = { ...defaults };
  let usedKnobs = false;
  let usedApplied = false;

  const knobs = Array.isArray(State?.runDetail?.knobs) ? State.runDetail.knobs : [];
  for(const r of knobs){
    const section = String(r?.section || "").trim().toLowerCase();
    const key = String(r?.key || "").trim();
    if(section !== "sim") continue;
    if(!keys.includes(key)) continue;
    values[key] = _parseKnobValue(r?.value);
    usedKnobs = true;
  }

  const appliedPayload = State?.runDetail?.appliedCommands;
  const itemsRaw = Array.isArray(appliedPayload?.items) ? appliedPayload.items : [];
  const items = itemsRaw.slice().sort((a,b)=>{
    const ga = Number(a?.applied_generation ?? a?.generation_applied ?? a?.target_generation ?? -1);
    const gb = Number(b?.applied_generation ?? b?.generation_applied ?? b?.target_generation ?? -1);
    return ga - gb;
  });
  for(const it of items){
    const st = String(it?.ack_status || "").toUpperCase();
    if(st && st !== "APPLIED" && st !== "ACCEPTED") continue;
    const payload = (it?.command_payload && typeof it.command_payload === "object")
      ? it.command_payload
      : ((it?.payload && typeof it.payload === "object") ? it.payload : it);
    const updates = Array.isArray(payload?.updates) ? payload.updates : [];
    for(const u of updates){
      const path = String(u?.path || "");
      const key = pathToKey.get(path);
      if(!key) continue;
      values[key] = u?.value;
      usedApplied = true;
    }
  }

  let source = "defaults";
  if(usedKnobs && usedApplied) source = "run_config + applied commands";
  else if(usedApplied) source = "applied commands";
  else if(usedKnobs) source = "run_config";

  return { values, source };
}

function buildPressureAutopilotPayload(){
  const enable = !!State.runDetail.pressureAutopilotEnable;
  // Default target gen should be near-future generation, not 1. Using 1 causes
  // immediate processing/rejection on any non-trivial run and makes the UI
  // look broken.
  const fallbackGen = (()=>{
    const cg = Number(State.runDetail?.meta?.run?.current_generation);
    return (Number.isFinite(cg) && cg >= 0) ? (cg + 10) : 10;
  })();
  const targetGen = _parseIntStrict(State.runDetail.pressureAutopilotTargetGen, fallbackGen);
  const interval = _parseIntStrict(State.runDetail.pressureAutopilotTickInterval, 200);
  const profile = String(State.runDetail.pressureAutopilotProfile || "balanced").trim().toLowerCase();
  const deltaUp = _parseFloatStrict(State.runDetail.pressureAutopilotDeltaUp, 0.05);
  const deltaDown = _parseFloatStrict(State.runDetail.pressureAutopilotDeltaDown, 0.10);
  const targetSurvival = _parseFloatStrict(State.runDetail.pressureAutopilotTargetSurvival, 0.55);
  const survivalCeiling = _parseFloatStrict(State.runDetail.pressureAutopilotSurvivalCeiling, 0.80);

  if(!Number.isFinite(targetGen) || targetGen < 0){
    throw new Error("Apply at gen must be >= 0");
  }
  if(enable){
    if(!Number.isFinite(interval) || interval < 1){
      throw new Error("Tick interval must be >= 1");
    }
    if(!["balanced","cycle","scarcity","cost","shock"].includes(profile)){
      throw new Error("Profile must be one of: balanced, cycle, scarcity, cost, shock");
    }
    if(!Number.isFinite(deltaUp) || deltaUp <= 0.0){
      throw new Error("delta_up must be > 0");
    }
    if(!Number.isFinite(deltaDown) || deltaDown <= 0.0){
      throw new Error("delta_down must be > 0");
    }
    if(!Number.isFinite(targetSurvival) || targetSurvival <= 0.0 || targetSurvival >= 1.0){
      throw new Error("target_survival must be in (0,1)");
    }
    if(!Number.isFinite(survivalCeiling) || survivalCeiling <= 0.0 || survivalCeiling >= 1.0){
      throw new Error("survival_ceiling must be in (0,1)");
    }
    return {
      schema_version: "cmd_v1",
      command_type: "PressureUpdateCommand",
      target_generation: targetGen,
      updates: [
        { path: "sim_cfg.pressure_autopilot_enable", value: true },
        { path: "sim_cfg.pressure_autopilot_tick_interval_gen", value: interval },
        { path: "sim_cfg.pressure_autopilot_profile", value: profile },
        { path: "sim_cfg.pressure_autopilot_delta_up", value: deltaUp },
        { path: "sim_cfg.pressure_autopilot_delta_down", value: deltaDown },
        { path: "sim_cfg.pressure_autopilot_target_survival", value: targetSurvival },
        { path: "sim_cfg.pressure_autopilot_survival_ceiling", value: survivalCeiling },
      ],
    };
  }

  return {
    schema_version: "cmd_v1",
    command_type: "PressureUpdateCommand",
    target_generation: targetGen,
    updates: [
      { path: "sim_cfg.pressure_autopilot_enable", value: false },
    ],
  };
}

function _builderSafeLinesToTokens(txt){
  const raw = String(txt||"")
    .split(/\r?\n/)
    .map(s=>String(s||"").trim())
    .filter(Boolean);
  // Support a single-line "--a 1 --b 2" paste too
  if(raw.length === 1 && raw[0].includes(" --")){
    return raw[0].split(/\s+/).map(s=>s.trim()).filter(Boolean);
  }
  return raw;
}

function _builderParseDefaultKnobs(txt){
  // Accept either:
  // - one kv per line: sim_cfg.foo=1
  // - or explicit tokens: --knob \n sim_cfg.foo=1
  const toks = _builderSafeLinesToTokens(txt);
  const out = [];
  for(let i=0;i<toks.length;i++){
    const t = String(toks[i]||"").trim();
    if(!t) continue;
    if(t === "--knob" && i+1 < toks.length){
      const kv = String(toks[i+1]||"").trim();
      if(kv) out.push(kv);
      i++;
      continue;
    }
    // Ignore other flags if someone pastes a whole CLI.
    if(t.startsWith("--")) continue;
    if(t.includes("=")) out.push(t);
  }
  // Unique while preserving order.
  const seen = new Set();
  const uniq = [];
  for(const kv of out){
    if(seen.has(kv)) continue;
    seen.add(kv);
    uniq.push(kv);
  }
  return uniq;
}

function _builderCheckboxOn(sel){
  return !!q(sel)?.checked;
}

function _builderStableJson(value){
  if(Array.isArray(value)){
    return "[" + value.map(_builderStableJson).join(",") + "]";
  }
  if(value && typeof value === "object"){
    const keys = Object.keys(value).sort();
    return "{" + keys.map(k=>JSON.stringify(k) + ":" + _builderStableJson(value[k])).join(",") + "}";
  }
  return JSON.stringify(value);
}

function _builderNormalizeAxisFlags(axisFlags){
  const out = {};
  const axes = Object.keys(axisFlags || {}).sort();
  for(const axis of axes){
    const mapping = axisFlags[axis];
    if(!mapping || typeof mapping !== "object") continue;
    const cleaned = {};
    for(const val of Object.keys(mapping).sort()){
      const toks = Array.isArray(mapping[val]) ? mapping[val].map(x=>String(x||"").trim()).filter(Boolean) : [];
      if(toks.length) cleaned[val] = toks;
    }
    if(Object.keys(cleaned).length) out[axis] = cleaned;
  }
  return out;
}

function _builderSetRegistrySelection(path){
  const next = String(path || "").trim();
  Builder.selectedRegistryPath = next;
  const sel = q("#rbRegistryLoadSelect");
  if(sel){
    try{ sel.value = next; }catch(_e){}
  }
}

function _builderActiveRegistryPath(){
  const sel = q("#rbRegistryLoadSelect");
  const current = String(sel?.value || Builder.selectedRegistryPath || "").trim();
  return current;
}

function _builderQuoteArg(s){
  const t = String(s ?? "");
  if(t === "") return "\"\"";
  if(/\s|\"/.test(t)){
    return "\"" + t.replace(/\\/g,"\\\\").replace(/\"/g,"\\\"") + "\"";
  }
  return t;
}

// -----------------------------
// Run Builder: determinism env prefix (PIS-7J2Q)
// -----------------------------

const _RB_LS_STRICT_KEY = "webui.sim_builder.strict_determinism";
const _RB_LS_LEGACY_KEY = "webui.sim_builder.legacy_debug_determinism";
const _RB_LS_PLAN_FREEZE_KEY = "webui.sim_builder.plan_freeze";

function _rbGetLs(key, fallback){
  try{
    const v = localStorage.getItem(String(key));
    return (v === null || v === undefined) ? String(fallback) : String(v);
  }catch(_e){
    return String(fallback);
  }
}

function _rbSetLs(key, value){
  try{ localStorage.setItem(String(key), String(value)); }catch(_e){}
}

function _builderStrictDeterminismOn(){
  const cb = q("#rbStrictDet");
  if(cb) return !!cb.checked;
  return _rbGetLs(_RB_LS_STRICT_KEY, "1") === "1";
}

function _builderLegacyDebugOn(){
  const cb = q("#rbLegacyDebugDet");
  if(cb) return !!cb.checked;
  return _rbGetLs(_RB_LS_LEGACY_KEY, "0") === "1";
}

function _builderPlanFreezeOn(){
  const cb = q("#rbPlanFreeze");
  if(cb) return !!cb.checked;
  return _rbGetLs(_RB_LS_PLAN_FREEZE_KEY, "0") === "1";
}

function _builderGetEnvPrefix(strictOn, legacyOn, planFreezeOn){
  // Stable ordering + no duplicates.
  const out = [];
  const seen = new Set();
  function add(kv){
    const key = String(kv).split("=", 1)[0];
    if(!key || seen.has(key)) return;
    seen.add(key);
    out.push(String(kv));
  }

  if(strictOn){
    add("PYTHONHASHSEED=0");
    add("EIEL_STRICT_DETERMINISM=1");
  }
  if(legacyOn){
    add("DEBUG_DETERMINISM=1");
  }
  if(planFreezeOn){
    add("EIEL_ENFORCE_PLAN_FREEZE=1");
  }
  return out.join(" ");
}

function _builderStripLeadingEnvKeys(cmd, keys){
  // Only strip at the very beginning. We don't try to be clever with quotes;
  // env prefixes are simple KEY=VALUE tokens.
  let s = String(cmd || "").trim();
  if(!s) return "";
  const want = new Set((keys || []).map(k=>String(k)));
  while(true){
    const m = s.match(/^([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)\s+([\s\S]*)$/);
    if(!m) break;
    const k = String(m[1]||"");
    if(!want.has(k)) break;
    s = String(m[3]||"").trim();
  }
  return s;
}

function _builderApplyEnvPrefix(cmd){
  const strictOn = _builderStrictDeterminismOn();
  const legacyOn = _builderLegacyDebugOn();
  const planFreezeOn = _builderPlanFreezeOn();
  const prefix = _builderGetEnvPrefix(strictOn, legacyOn, planFreezeOn);
  const keys = ["PYTHONHASHSEED", "EIEL_STRICT_DETERMINISM", "DEBUG_DETERMINISM", "EIEL_ENFORCE_PLAN_FREEZE"];
  const base = _builderStripLeadingEnvKeys(cmd, keys);
  if(!prefix) return base;
  if(!base) return prefix;
  return `${prefix} ${base}`;
}

function _builderHarnessReadinessKey(cmd){
  const keys = ["PYTHONHASHSEED", "EIEL_STRICT_DETERMINISM", "DEBUG_DETERMINISM", "EIEL_ENFORCE_PLAN_FREEZE"];
  const base = _builderStripLeadingEnvKeys(cmd, keys);
  const direct = String(base || "").trim();
  if(!direct) return "";
  const wrapped = direct.match(/^python\s+-m\s+tools\.run_job\b([\s\S]*)$/);
  if(wrapped){
    const mm = String(wrapped[1] || "").match(/(?:^|\s)--module\s+([^\s]+)/);
    const mod = String(mm?.[1] || "").trim();
    return mod || direct;
  }
  const mm = direct.match(/^python\s+-m\s+([^\s]+)/);
  const mod = String(mm?.[1] || "").trim();
  return mod || direct;
}

function _builderGuessRunsRoot(){
  // Prefer the active builder UI field (harness runs_root), then CLI runs_dir.
  // Fall back to server defaults.
  const a = String(q("#rbHarnessRunsRoot")?.value || "").trim();
  if(a) return a;
  const b = String(q("#rbRunsDir")?.value || "").trim();
  if(b) return b;
  const c = String((Builder && Builder.cfg && Builder.cfg.builder && Builder.cfg.builder.runs_root_default) ? Builder.cfg.builder.runs_root_default : "").trim();
  if(c) return c;
  return "experiments_data/emergence/runs";
}

function _builderHarnessModulePrefix(){
  return String(Builder?.cfg?.builder?.harness_module_prefix || "experiments.harnesses.generated").trim();
}

function _builderRunJobModule(){
  return String(Builder?.cfg?.builder?.run_job_module || "experiments_shared.run_job").trim();
}

function _builderManifestRunnerModule(){
  return String(Builder?.cfg?.builder?.manifest_runner_module || "experiments_shared.manifest_runner").trim();
}

function _builderWrapperRootFromRunsRoot(runsRoot){
  const root = String(runsRoot || "").trim() || "experiments_data/emergence/runs";
  return root.replace(/\/+$/, "") + "/wrapper_jobs";
}

function _builderRegistryLabel(r){
  const uid = String(r?.uid || r?.canonical_id || "").trim();
  const path = String(r?.path || "").trim();
  const generated = path.match(/__([A-Za-z0-9]{5,})__(?:harness|registry)?\.ya?ml$/i);
  const stem = path.split("/").pop() || "";
  const stemSuffix = stem.match(/__([A-Za-z0-9]{5,})__(?:harness)?(?:\.ya?ml)?$/i);
  const suffix = String(generated?.[1] || stemSuffix?.[1] || r?.display_suffix || "").trim();
  if(uid && suffix) return `${uid} · ${suffix}`;
  if(uid) return uid;
  return displayId(r, {fallback: String(r?.path || "")});
}

// Minimal self-tests (PIS-7J2Q AC6): runs in-browser and reports to console.
// This is not a full test suite, but it does catch regressions in the two
// string-manipulation contracts we care about: env prefix + wrapper_root.
function _builderRunSelfTests(){
  try{
    const errs = [];

    // Env prefix ordering and no duplicates.
    const p1 = _builderGetEnvPrefix(true, true, true);
    if(p1 !== "PYTHONHASHSEED=0 EIEL_STRICT_DETERMINISM=1 DEBUG_DETERMINISM=1 EIEL_ENFORCE_PLAN_FREEZE=1"){
      errs.push("env_prefix ordering mismatch");
    }
    const p2 = _builderGetEnvPrefix(true, false, false);
    if(p2 !== "PYTHONHASHSEED=0 EIEL_STRICT_DETERMINISM=1"){
      errs.push("env_prefix strict-only mismatch");
    }
    const p3 = _builderGetEnvPrefix(false, true, false);
    if(p3 !== "DEBUG_DETERMINISM=1"){
      errs.push("env_prefix legacy-only mismatch");
    }
    const p4 = _builderGetEnvPrefix(false, false, true);
    if(p4 !== "EIEL_ENFORCE_PLAN_FREEZE=1"){
      errs.push("env_prefix plan-freeze mismatch");
    }

    // wrapper_root injection for managed.
    const rr = "experiments_data/emergence/runs";
    const cmd = `python -m ${_builderHarnessModulePrefix()}.foo__harness`;
    // Pretend the UI has runs_root set.
    const wr = _builderWrapperRootFromRunsRoot(rr);
    const managed = _builderCmdToManaged(cmd);
    if(!managed.includes("--wrapper_root") || !managed.includes("wrapper_jobs")){
      errs.push("managed command missing wrapper_root");
    }

    if(errs.length){
      console.error("[run_builder_selftest] FAIL", errs);
    }else{
      console.log("[run_builder_selftest] PASS");
    }
  }catch(e){
    try{ console.error("[run_builder_selftest] ERROR", e); }catch(_e){}
  }
}

function _builderCmdToManaged(cmd){
  const s = String(cmd || "").trim();
  const runJobModule = _builderRunJobModule();
  if(!s) return "";
  // If it's already wrapped, keep it.
  if(new RegExp(`^python\\s+-m\\s+(${runJobModule.replace(/\./g, "\\.")}|tools\\.run_job)\\b`).test(s)){
    // Ensure wrapper_root is present (PIS-7QK2).
    if(/\s--wrapper_root\b/.test(s)) return s;
    const wr = _builderWrapperRootFromRunsRoot(_builderGuessRunsRoot());
    return s + ` --wrapper_root ${_builderQuoteArg(wr)}`;
  }

  // Basic parse: python -m <module> <args...>
  const m = s.match(/^python\s+-m\s+([^\s]+)([\s\S]*)$/);
  if(!m) return s;
  const mod = m[1];
  const rest = String(m[2] || "").trim();

  // ---- Required run_job fields (best-effort) ----
  // We prefer the server-provided builder config git_head when available.
  const gitRef = String((Builder && Builder.cfg && Builder.cfg.git_head) ? Builder.cfg.git_head : "").trim();

  // Seed: use first token from Harness seeds field if present; fall back to 1.
  let seed = 1;
  try{
    const seedsRaw = String(q("#rbHarnessSeeds")?.value || "").trim();
    if(seedsRaw){
      const tok = seedsRaw.split(/[^0-9-]+/).filter(Boolean)[0];
      const n = parseInt(tok, 10);
      if(Number.isFinite(n)) seed = n;
    }
  }catch(_e){}

  // job_id: derive from module name (stable and human-readable).
  // - If module ends with '__harness', use that suffix.
  // - Otherwise fall back to the module basename.
  let jobId = "";
  try{
    const base = String(mod.split(".").slice(-1)[0] || "").trim();
    jobId = base || "job";
    // Keep job_id filesystem-friendly.
    jobId = jobId.replace(/[^A-Za-z0-9_\-]+/g, "_");
    if(jobId.length > 96) jobId = jobId.slice(0, 96);
  }catch(_e){
    jobId = "job";
  }

  const wrapperRoot = _builderWrapperRootFromRunsRoot(_builderGuessRunsRoot());

  // If we don't have a git ref, keep the wrapper in a "best-effort" mode.
  // (It may still fail at runtime, but it's better than inventing a fake SHA.)
  if(!gitRef){
    return rest
      ? `python -m ${runJobModule} --module ${mod} --wrapper_root ${_builderQuoteArg(wrapperRoot)} -- ${rest}`
      : `python -m ${runJobModule} --module ${mod} --wrapper_root ${_builderQuoteArg(wrapperRoot)}`;
  }

  // Use '--' to separate run_job args from the wrapped module args.
  return rest
    ? `python -m ${runJobModule} --module ${mod} --git_ref ${gitRef} --job_id ${jobId} --seed ${seed} --wrapper_root ${_builderQuoteArg(wrapperRoot)} -- ${rest}`
    : `python -m ${runJobModule} --module ${mod} --git_ref ${gitRef} --job_id ${jobId} --seed ${seed} --wrapper_root ${_builderQuoteArg(wrapperRoot)}`;
}
function _builderBuildCliCommand(){
  const entry = q("#rbEntrypoint");
  const mod = entry ? String(entry.value || "engine.main") : "engine.main";

  const pairs = [
    ["--runs_dir", q("#rbRunsDir")?.value],
    ["--run_id", q("#rbRunId")?.value],
    ["--run_tag", q("#rbRunTag")?.value],
    ["--resume_from", q("#rbResume")?.value],
    ["--generations", q("#rbGenerations")?.value],
    ["--steps", q("#rbSteps")?.value],
    ["--seed", q("#rbSeed")?.value],
    ["--pop", q("#rbPop")?.value],
    ["--grid_size", q("#rbGrid")?.value],
    ["--policy_tick_stride", q("#rbStride")?.value],
    ["--checkpoint_period_gen", q("#rbCkptEvery")?.value],
    ["--checkpoint_rotate", q("#rbCkptKeep")?.value],
    ["--device", q("#rbDevice")?.value],
  ];

  const args = [];
  for(const [k,v] of pairs){
    const t = String(v ?? "").trim();
    if(!t) continue;
    args.push(k);
    args.push(_builderQuoteArg(t));
  }

  const rawArgs = _builderSafeLinesToTokens(q("#rbRawArgs")?.value);
  for(const tok of rawArgs){
    args.push(_builderQuoteArg(tok));
  }

  const cmd = ["python", "-m", mod, ...args].join(" ");
  return cmd;
}

function _builderRenderCliPreview(){
  const out = q("#rbCmdOut");
  if(!out) return;
  try{
    const raw = _builderBuildCliCommand();
    Builder.lastCliCmdRaw = raw;
    _builderWriteCommandPreview(out, _builderApplyEnvPrefix(_builderCmdToManaged(raw)));
  }catch(e){
    _builderWriteTextLike(out, String(e?.message||e));
  }
  _builderApplyStartButtonState();
}

function _builderCollectAxesFromDom(){
  const rows = qa(".rb-axis-row");
  const axes = [];
  for(const row of rows){
    const nameEl = q(".rb-axis-name", row);
    const valsTxt = q(".rb-axis-values", row)?.value;
    const axisName = nameEl
      ? String(nameEl.value || "").trim()
      : String(row.getAttribute("data-axis-name") || "").trim();
    if(!axisName) continue;
    const vals = String(valsTxt||"")
      .split(/\r?\n/)
      .map(s=>s.trim())
      .filter(Boolean);
    axes.push({name: axisName, values: vals});
  }
  return axes;
}

function _builderAxisKey(axisName){
  return String(axisName||"").trim();
}

function _builderEnsureAxisFlags(axisName){
  const k = _builderAxisKey(axisName);
  if(!Builder.axisFlags[k]) Builder.axisFlags[k] = {};
  return Builder.axisFlags[k];
}

function _builderCaseValuesFromAxes(axes){
  for(const ax of axes){
    if(String(ax.name) === "case"){
      return Array.isArray(ax.values) ? ax.values : [];
    }
  }
  return [];
}

function _builderRefreshCaseDropdown(){
  const sel = q("#rbCaseSelect");
  if(!sel) return;
  const axes = _builderCollectAxesFromDom();
  const cases = _builderCaseValuesFromAxes(axes);
  const cur = String(sel.value||"");
  sel.innerHTML = "";
  for(const c of cases){
    const opt = document.createElement("option");
    opt.value = String(c);
    opt.textContent = String(c);
    sel.appendChild(opt);
  }
  if(cases.includes(cur)) sel.value = cur;
}

function _builderRenderFlagsForSelectedCase(){
  const axisName = "case";
  const sel = q("#rbCaseSelect");
  const out = q("#rbCaseFlagsOut");
  if(!sel || !out) return;
  const caseVal = String(sel.value||"").trim();
  const flags = (_builderEnsureAxisFlags(axisName)[caseVal] || []);
  out.value = Array.isArray(flags) ? flags.join("\n") : "";
}

function _builderUpdateFlagsFromEditor(){
  const axisName = "case";
  const sel = q("#rbCaseSelect");
  const out = q("#rbCaseFlagsOut");
  if(!sel || !out) return;
  const caseVal = String(sel.value||"").trim();
  const toks = _builderSafeLinesToTokens(out.value);
  _builderEnsureAxisFlags(axisName)[caseVal] = toks;
  _builderRefreshHarnessDirtyState();
  _builderPersistSoon();
}

function _builderAddKnobToCase(){
  const sel = q("#rbCaseSelect");
  const knobKey = q("#rbKnobKey");
  const knobVal = q("#rbKnobVal");
  if(!sel || !knobKey || !knobVal) return;
  const caseVal = String(sel.value||"").trim();
  const k = String(knobKey.value||"").trim();
  const v = String(knobVal.value||"").trim();
  if(!caseVal || !k) return;
  const kv = (k.includes("=") && !v) ? k : `${k}=${v}`;
  if(!kv.includes("=")){
    showToast("Knob value missing", "Provide a value (e.g. true / 4.0)");
    return;
  }
  const axisMap = _builderEnsureAxisFlags("case");
  const cur = Array.isArray(axisMap[caseVal]) ? axisMap[caseVal] : [];
  axisMap[caseVal] = cur.concat(["--knob", kv]);
  knobKey.value = "";
  knobVal.value = "";
  _builderRenderFlagsForSelectedCase();
  _builderRefreshHarnessDirtyState();
  _builderPersistSoon();
}

function _builderAddRawFlagToCase(){
  const sel = q("#rbCaseSelect");
  const flag = q("#rbRawFlag");
  if(!sel || !flag) return;
  const caseVal = String(sel.value||"").trim();
  const tok = String(flag.value||"").trim();
  if(!caseVal || !tok) return;
  const axisMap = _builderEnsureAxisFlags("case");
  const cur = Array.isArray(axisMap[caseVal]) ? axisMap[caseVal] : [];
  axisMap[caseVal] = cur.concat([tok]);
  flag.value = "";
  _builderRenderFlagsForSelectedCase();
  _builderRefreshHarnessDirtyState();
  _builderPersistSoon();
}

function _builderBuildHarnessPayload(){
  const testId = String(q("#rbHarnessName")?.value||"").trim();
  const desc = String(q("#rbHarnessDesc")?.value||"").trim();
  const question = String(q("#rbHarnessQuestion")?.value||"").trim();
  const runsRoot = String(q("#rbHarnessRunsRoot")?.value||"").trim();

  const seeds = csvListToArray(q("#rbHarnessSeeds")?.value).map(x=>parseInt(x,10)).filter(Number.isFinite);

  const axes = _builderCollectAxesFromDom();
  const expectedArtifacts = _builderSafeLinesToTokens(q("#rbHarnessArtifacts")?.value);
  const defaultKnobs = _builderParseDefaultKnobs(q("#rbHarnessDefaultKnobs")?.value);

  // CLI defaults (basic)
  const cliDefaults = {
    generations: maybeInt(q("#rbHarnessGenerations")?.value),
    steps: maybeInt(q("#rbHarnessSteps")?.value),
    pop: maybeInt(q("#rbHarnessPop")?.value),
    policy_tick_stride: maybeInt(q("#rbHarnessStride")?.value),
    grid_size: maybeInt(q("#rbHarnessGrid")?.value),
    checkpoint_period_gen: maybeInt(q("#rbHarnessCkptEvery")?.value),
    checkpoint_rotate: maybeInt(q("#rbHarnessCkptKeep")?.value),
  };

  // axis flags: only send non-empty
  const axisFlags = {};
  for(const [ax, mapping] of Object.entries(Builder.axisFlags||{})){
    if(!mapping || typeof mapping !== "object") continue;
    const out = {};
    for(const [val, toks] of Object.entries(mapping)){
      if(Array.isArray(toks) && toks.length){ out[val] = toks; }
    }
    if(Object.keys(out).length){ axisFlags[ax] = out; }
  }

  return {
    test_id: testId,
    description: desc,
    intent_question: question,
    runs_root: runsRoot,
    seeds: seeds,
    axes: axes,
    axis_flags: axisFlags,
    default_knobs: defaultKnobs,
    perf: _builderCheckboxOn("#rbHarnessPerf"),
    enable_profiler: _builderCheckboxOn("#rbHarnessProfiler"),
    cli_defaults: cliDefaults,
    expected_artifacts: expectedArtifacts,
  };
}

function _builderHarnessSignatureForPayload(payload){
  if(!payload || typeof payload !== "object") return "";
  const cliDefaults = (payload.cli_defaults && typeof payload.cli_defaults === "object") ? payload.cli_defaults : {};
  const normalized = {
    test_id: String(payload.test_id || "").trim(),
    description: String(payload.description || "").trim(),
    intent_question: String(payload.intent_question || "").trim(),
    runs_root: String(payload.runs_root || "").trim(),
    seeds: Array.isArray(payload.seeds) ? payload.seeds.map(x=>parseInt(x, 10)).filter(Number.isFinite) : [],
    axes: Array.isArray(payload.axes)
      ? payload.axes.map(ax=>({
          name: String(ax?.name || "").trim(),
          values: Array.isArray(ax?.values) ? ax.values.map(v=>String(v || "").trim()).filter(Boolean) : [],
        }))
      : [],
    axis_flags: _builderNormalizeAxisFlags(payload.axis_flags || {}),
    default_knobs: Array.isArray(payload.default_knobs) ? payload.default_knobs.map(v=>String(v || "").trim()).filter(Boolean) : [],
    perf: !!payload.perf,
    enable_profiler: !!payload.enable_profiler,
    cli_defaults: {
      generations: cliDefaults.generations ?? null,
      steps: cliDefaults.steps ?? null,
      pop: cliDefaults.pop ?? null,
      policy_tick_stride: cliDefaults.policy_tick_stride ?? null,
      grid_size: cliDefaults.grid_size ?? null,
      checkpoint_period_gen: cliDefaults.checkpoint_period_gen ?? null,
      checkpoint_rotate: cliDefaults.checkpoint_rotate ?? null,
    },
    expected_artifacts: Array.isArray(payload.expected_artifacts)
      ? payload.expected_artifacts.map(v=>String(v || "").trim()).filter(Boolean)
      : [],
  };
  return _builderStableJson(normalized);
}

function _builderCurrentHarnessSignature(){
  try{
    return _builderHarnessSignatureForPayload(_builderBuildHarnessPayload());
  }catch(_e){
    return "";
  }
}

function _builderSetHarnessBaseline(path, harnessPath, harnessExists){
  Builder.loadedRegistryPath = String(path || "").trim();
  Builder.selectedRegistryPath = String(path || "").trim();
  Builder.loadedHarnessPath = String(harnessPath || "").trim();
  Builder.loadedHarnessExists = !!harnessExists;
  Builder.harnessBaselineSignature = _builderCurrentHarnessSignature();
  Builder.harnessDirty = false;
}

function _builderClearHarnessBaseline(){
  Builder.loadedRegistryPath = "";
  Builder.selectedRegistryPath = "";
  Builder.loadedHarnessPath = "";
  Builder.loadedHarnessExists = false;
  Builder.harnessBaselineSignature = "";
  Builder.harnessDirty = true;
}

function _builderRefreshRegistryActionState(){
  const delBtn = q("#rbRegistryDeleteBtn");
  if(delBtn){
    const regs = Array.isArray(Builder.registries) ? Builder.registries : [];
    delBtn.disabled = !regs.length;
    delBtn.title = regs.length ? "Select registries to delete" : "No registries available";
  }
}

function _builderRefreshHarnessDirtyState(){
  const btn = q("#rbHarnessGen");
  const box = q("#rbHarnessDirtyState");
  let payload = null;
  try{
    payload = _builderBuildHarnessPayload();
  }catch(_e){
    payload = null;
  }
  const sig = payload ? _builderHarnessSignatureForPayload(payload) : "";
  const problems = [];
  const testId = String(payload?.test_id || "").trim();
  if(!testId) problems.push("name required");
  const seeds = Array.isArray(payload?.seeds) ? payload.seeds : [];
  if(!seeds.length) problems.push("seed required");
  const axes = Array.isArray(payload?.axes) ? payload.axes : [];
  const hasAxisValues = axes.some(ax=>Array.isArray(ax?.values) && ax.values.some(v=>String(v || "").trim()));
  if(!hasAxisValues) problems.push("add at least one axis value");

  const hasBaseline = !!String(Builder.harnessBaselineSignature || "").trim();
  const hasGeneratedTarget = !!Builder.loadedHarnessExists;
  const dirty = !hasBaseline || sig !== Builder.harnessBaselineSignature;
  Builder.harnessDirty = !!dirty;

  let label = "Ready";
  if(problems.length){
    label = `Incomplete: ${problems.join(" · ")}`;
  }else if(hasBaseline && !dirty && hasGeneratedTarget){
    label = "Up to date";
  }else if(hasBaseline && !dirty){
    label = "YAML loaded; harness missing";
  }else if(hasBaseline){
    label = "Modified";
  }
  if(box) box.textContent = label;

  if(btn){
    const blocked = problems.length > 0 || (hasBaseline && !dirty && hasGeneratedTarget);
    btn.disabled = !!blocked;
    if(problems.length){
      btn.title = problems.join(" · ");
    }else if(hasBaseline && !dirty && hasGeneratedTarget){
      btn.title = "Nothing changed since the current generated registry";
    }else{
      btn.title = "Generate harness + registry";
    }
  }
  _builderApplyStartButtonState();
}

function _builderApplyHarnessPayloadToForm(hp){
  if(!hp || typeof hp !== "object") return;

  // Basics
  if(q("#rbHarnessName")) q("#rbHarnessName").value = hp.test_id || "";
  if(q("#rbHarnessDesc")) q("#rbHarnessDesc").value = hp.description || "";
  if(q("#rbHarnessQuestion")) q("#rbHarnessQuestion").value = hp.intent_question || "";
  if(q("#rbHarnessDefaultKnobs")) q("#rbHarnessDefaultKnobs").value = Array.isArray(hp.default_knobs) ? hp.default_knobs.join("\n") : "";
  if(q("#rbHarnessPerf")) q("#rbHarnessPerf").checked = !!hp.perf;
  if(q("#rbHarnessProfiler")) q("#rbHarnessProfiler").checked = !!hp.enable_profiler;

  // Seeds
  if(q("#rbHarnessSeeds") && Array.isArray(hp.seeds)){
    q("#rbHarnessSeeds").value = hp.seeds.map(x=>String(x)).join(",");
  }

  // Expected artifacts
  if(q("#rbHarnessArtifacts") && Array.isArray(hp.expected_artifacts)){
    q("#rbHarnessArtifacts").value = hp.expected_artifacts.join("\n");
  }

  // CLI defaults
  const cd = (hp.cli_defaults && typeof hp.cli_defaults === "object") ? hp.cli_defaults : {};
  if(q("#rbHarnessGenerations") && cd.generations != null) q("#rbHarnessGenerations").value = cd.generations;
  if(q("#rbHarnessSteps") && cd.steps != null) q("#rbHarnessSteps").value = cd.steps;
  if(q("#rbHarnessPop") && cd.pop != null) q("#rbHarnessPop").value = cd.pop;
  if(q("#rbHarnessGrid") && cd.grid_size != null) q("#rbHarnessGrid").value = cd.grid_size;
  if(q("#rbHarnessStride") && cd.policy_tick_stride != null) q("#rbHarnessStride").value = cd.policy_tick_stride;
  if(q("#rbHarnessCkptEvery") && cd.checkpoint_period_gen != null) q("#rbHarnessCkptEvery").value = cd.checkpoint_period_gen;
  if(q("#rbHarnessCkptKeep") && cd.checkpoint_rotate != null) q("#rbHarnessCkptKeep").value = cd.checkpoint_rotate;

  // Axes replace
  const axesBox = q("#rbAxes");
  if(axesBox){ axesBox.innerHTML = ""; }
  const axes = Array.isArray(hp.axes) ? hp.axes : [];
  if(axes.length){
    for(const ax of axes){
      _builderAddAxisRow(ax?.name||"axis", Array.isArray(ax?.values)?ax.values:[]);
    }
  }else{
    _builderAddAxisRow("case", []);
  }

  // Axis flags
  Builder.axisFlags = (hp.axis_flags && typeof hp.axis_flags === "object") ? hp.axis_flags : {};
  _builderRefreshCaseDropdown();
  _builderRenderFlagsForSelectedCase();

  _builderRenderHarnessCmdPreview();
  _builderRefreshHarnessDirtyState();
  _builderPersistNow();
}

function _builderBuildManifestPayload(){
  const testName = String(q("#rbManifestName")?.value||"").trim();
  const strict = !!q("#rbManifestStrict")?.checked;
  const outRoot = String(q("#rbManifestOutRoot")?.value||"").trim();
  const gitRefOverride = String(q("#rbManifestGitRef")?.value||"").trim();
  const regsChecked = qa(".rb-reg-item input[type=checkbox]:checked").map(cb=>String(cb.value||"").trim()).filter(Boolean);
  const regsManual = _builderSafeLinesToTokens(q("#rbManifestManual")?.value);
  const regs = regsChecked.concat(regsManual).filter(Boolean);
  return {
    testname: testName,
    registry_paths: regs,
    output_root: outRoot,
    strict: strict,
    git_ref_override: gitRefOverride,
  };
}

async function _builderLoadRegistryIntoEditor(path){
  const p = String(path || "").trim();
  const st = q("#rbRegistryLoadStatus");
  if(!p){
    if(st) st.textContent = "";
    return false;
  }
  if(st) st.textContent = "loading...";
  try{
    const res = await apiBuilderRegistryLoad({path: p});
    const hp = res?.harness_payload;
    _builderApplyHarnessPayloadToForm(hp);
    const sourcePath = String(res?.source_path || p).trim();
    const harnessExists = !!res?.harness_exists;
    const harnessPath = String(res?.harness_path || "").trim();
    _builderSetRegistrySelection(sourcePath);
    _builderSetHarnessBaseline(sourcePath, harnessPath, harnessExists);
    if(res?.run_command){
      Builder.lastHarnessCmdRaw = String(res.run_command || "");
      const cmdOut = q("#rbHarnessCmdOut");
      _builderWriteCommandPreview(cmdOut, _builderApplyEnvPrefix(_builderCmdToManaged(String(res.run_command_managed || res.run_command || ""))));
    }else{
      _builderRenderHarnessCmdPreview();
    }
    Builder.harnessLaunchReady = !!harnessExists;
    _builderApplyStartButtonState();
    _builderRefreshHarnessDirtyState();
    _builderRefreshRegistryActionState();
    if(st) st.textContent = harnessExists ? `loaded: ${sourcePath}` : `loaded: ${sourcePath} (generate to create harness)`;
    showToast("Loaded", sourcePath);
    return true;
  }catch(e){
    if(st) st.textContent = "load failed";
    showToast("Load failed", e?.message||String(e));
    return false;
  }
}

function _builderRenderRegistryList(){
  const box = q("#rbRegistries");
  if(!box) return;
  const regs = Array.isArray(Builder.registries) ? Builder.registries : [];
  if(!regs.length){
    box.innerHTML = `<div class="muted">No registries found (yet). Generated ones will appear here once created.</div>`;
    _builderRenderRegistryLoadSelect();
    return;
  }
  const rows = regs.map(r=>{
    const p = escapeHtml(r.path || "");
    const shown = escapeHtml(_builderRegistryLabel(r));
    const meta = r.mtime_utc ? ` <span class="muted">(${escapeHtml(r.mtime_utc)})</span>` : "";
    const tag = r.harness_exists ? `<span class="badge badge-accent">harness</span>` : `<span class="badge badge-muted">yaml</span>`;
    return `<div class="rb-reg-item" title="${p}">
      <span class="rb-reg-copy">
        <span class="rb-reg-title">${shown}</span>
        <span class="rb-reg-path">${p}</span>
      </span>
      ${meta} ${tag}
    </div>`;
  }).join("");
  box.innerHTML = rows;
  _builderRenderRegistryLoadSelect();
}

function _builderRenderRegistryLoadSelect(){
  const sel = q("#rbRegistryLoadSelect");
  if(!sel) return;
  const regs = Array.isArray(Builder.registries) ? Builder.registries : [];
  const cur = _builderActiveRegistryPath();
  sel.innerHTML = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "(select registry)";
  sel.appendChild(empty);
  for(const r of regs){
    const opt = document.createElement("option");
    opt.value = String(r.path||"");
    opt.textContent = _builderRegistryLabel(r);
    opt.title = String(r.path || "");
    sel.appendChild(opt);
  }
  if(cur) sel.value = cur;
  _builderRefreshRegistryActionState();
}

async function _builderLoadRegistrySelectionWithGuard(path){
  const next = String(path || "").trim();
  if(!next){
    _builderSetRegistrySelection("");
    _builderRefreshRegistryActionState();
    return false;
  }
  const loaded = String(Builder.loadedRegistryPath || "").trim();
  if(loaded && next !== loaded && Builder.harnessDirty){
    const ok = window.confirm(`Discard current builder edits and load:\n\n${next}`);
    if(!ok){
      _builderSetRegistrySelection(loaded);
      _builderRefreshRegistryActionState();
      return false;
    }
  }
  return await _builderLoadRegistryIntoEditor(next);
}

async function _builderDeleteRegistryPaths(paths){
  if(!paths.length){
    showToast("No registries selected", "Select one or more registries to delete");
    return false;
  }
  const previews = [];
  for(const path of paths){
    previews.push(await apiBuilderRegistryDeletePreview({path}));
  }
  const deleteGroups = previews.map((preview, idx)=>{
    const files = Array.isArray(preview?.delete_paths) ? preview.delete_paths.map(x=>String(x || "").trim()).filter(Boolean) : [];
    return { path: paths[idx], files };
  }).filter((it)=>it.files.length);
  if(!deleteGroups.length){
    showToast("Delete failed", "No files resolved for deletion");
    return false;
  }
  const summary = deleteGroups.map((it)=>`${it.path}\n  ${it.files.join("\n  ")}`).join("\n\n");
  const ok = window.confirm(`Delete these registries?\n\n${summary}`);
  if(!ok) return false;

  for(const it of deleteGroups){
    await apiBuilderRegistryDelete({path: it.path, confirm_paths: it.files});
    if(String(Builder.loadedRegistryPath || "").trim() === it.path){
      _builderClearHarnessBaseline();
      Builder.harnessLaunchReady = false;
      _builderSetRegistrySelection("");
      _builderRenderHarnessCmdPreview();
    }
  }
  await _builderRefreshLists();
  _builderRenderRegistryList();
  _builderRefreshHarnessDirtyState();
  _builderRefreshRegistryActionState();
  const st = q("#rbRegistryLoadStatus");
  if(st) st.textContent = `deleted: ${deleteGroups.length} registr${deleteGroups.length === 1 ? "y" : "ies"}`;
  showToast("Deleted", `${deleteGroups.length} registr${deleteGroups.length === 1 ? "y" : "ies"} removed`);
  return true;
}

function _builderOpenDeleteRegistriesModal(){
  const regs = Array.isArray(Builder.registries) ? Builder.registries : [];
  if(!regs.length){
    showToast("No registries", "No registries available to delete");
    return;
  }
  const body = `
    <div class="muted" style="margin-bottom:10px;">Select one or more registries to delete.</div>
    <div id="rbDeleteRegistryList" class="rb-reg-list">
      ${regs.map((r, idx)=>{
        const path = escapeHtml(String(r.path || ""));
        const shown = escapeHtml(displayId(r, {fallback: String(r.path || "")}));
        const meta = r.mtime_utc ? ` <span class="muted">(${escapeHtml(r.mtime_utc)})</span>` : "";
        return `<label class="rb-reg-item" title="${path}">
          <input class="rb-del-reg-check" type="checkbox" value="${path}" ${idx === 0 ? "checked" : ""}/>
          <span class="rb-reg-copy">
            <span class="rb-reg-title">${shown}</span>
            <span class="rb-reg-path">${path}</span>
          </span>
          ${meta}
        </label>`;
      }).join("")}
    </div>
    <div class="row rb-inline-actions" style="margin-top:12px;justify-content:flex-end;">
      <button id="rbDeleteRegistryCancel" class="btn small" type="button">Cancel</button>
      <button id="rbDeleteRegistryConfirm" class="btn danger small" type="button">Delete selected</button>
    </div>
  `;
  openModal("Delete registries", body, (scope)=>{
    safeOn(q("#rbDeleteRegistryCancel", scope), "click", ()=>closeModal());
    safeOn(q("#rbDeleteRegistryConfirm", scope), "click", async ()=>{
      const selected = qa(".rb-del-reg-check:checked", scope)
        .map((it)=>String(it?.value || "").trim())
        .filter(Boolean);
      if(!selected.length){
        showToast("No registries selected", "Check at least one registry");
        return;
      }
      try{
        await _builderDeleteRegistryPaths(selected);
        closeModal();
      }catch(e){
        showToast("Delete failed", e?.message || String(e));
      }
    });
  });
}


async function _builderEnsureRunsLoaded(){
  if(Array.isArray(State.runs) && State.runs.length) return;
  try{
    const runs = await apiRuns();
    State.runs = Array.isArray(runs) ? runs : [];
  }catch(e){
    State.runs = State.runs || [];
  }
}

function _builderRenderResumePicker(){
  const selRun = q("#rbResumeRun");
  const selCk = q("#rbResumeCkpt");
  if(!selRun || !selCk) return;

  const runs = Array.isArray(State.runs) ? State.runs : [];
  const sorted = runs.slice().sort((a,b)=>{
    const ta = Date.parse(a?.updated_utc || a?.created_utc || "") || 0;
    const tb = Date.parse(b?.updated_utc || b?.created_utc || "") || 0;
    return tb - ta;
  });

  const cur = String(selRun.value || Builder._resumePickWanted?.run || "");
  let html = `<option value="">(select run)</option>`;
  for(const r of sorted.slice(0, 2000)){
    const rk = runKeyOf(r);
    if(!rk) continue;
    const label = `${rk}${r?.label ? " · " + r.label : ""}`;
    html += `<option value="${escapeHtml(rk)}">${escapeHtml(label)}</option>`;
  }
  selRun.innerHTML = html;

  if(cur){
    selRun.value = cur;
  }

  if(String(selRun.value||"").trim()){
    _builderRefreshResumeCkpts();
  }else{
    selCk.innerHTML = `<option value="">(select checkpoint)</option>`;
  }
}

async function _builderRefreshResumeCkpts(){
  const selRun = q("#rbResumeRun");
  const selCk = q("#rbResumeCkpt");
  if(!selRun || !selCk) return;

  const rk = String(selRun.value||"").trim();
  if(!rk){
    selCk.innerHTML = `<option value="">(select checkpoint)</option>`;
    return;
  }

  selCk.innerHTML = `<option value="">Loading…</option>`;
  try{
    const payload = await apiRunCheckpoints(rk);
    const cps = Array.isArray(payload?.checkpoints) ? payload.checkpoints : [];
    cps.sort((a,b)=>(Number(a?.generation)||0)-(Number(b?.generation)||0));
    let html = `<option value="">(select checkpoint)</option>`;
    for(const c of cps){
      const rel = String(c?.relative_path||"");
      if(!rel) continue;
      const gen = Number(c?.generation);
      const id = String(c?.checkpoint_id||"");
      const exists = !!c?.exists;
      const label = `${Number.isFinite(gen)?("gen "+gen):""} ${id}${exists?"":" (missing)"}`;
      html += `<option value="${escapeHtml(rel)}">${escapeHtml(label.trim())}</option>`;
    }
    selCk.innerHTML = html;

    const wanted = Builder._resumePickWanted?.ckpt ? String(Builder._resumePickWanted.ckpt) : "";
    if(wanted){
      selCk.value = wanted;
      Builder._resumePickWanted = null;
    }
  }catch(e){
    selCk.innerHTML = `<option value="">(failed)</option>`;
    showToast("Resume picker", e?.message||String(e));
  }
}

function _builderRenderEntrypoints(){
  const sel = q("#rbEntrypoint");
  if(!sel) return;
  const eps = Array.isArray(Builder.cfg?.entrypoints) ? Builder.cfg.entrypoints : [];
  const cur = String(sel.value||"");
  sel.innerHTML = "";
  for(const e of eps){
    const opt = document.createElement("option");
    opt.value = String(e.module||"");
    opt.textContent = String(e.label||e.module||"");
    sel.appendChild(opt);
  }
  // Fallback if server didn't provide anything.
  if(!eps.length){
    const opt = document.createElement("option");
    opt.value = "engine.main";
    opt.textContent = "engine.main";
    sel.appendChild(opt);
  }
  if(cur){
    try{ sel.value = cur; }catch(_e){}
  }
}

function _builderRenderKnobKeyDatalist(){
  const sel = q("#rbKnobKey");
  if(!sel) return;
  const keys = Array.isArray(Builder.knobKeys) ? Builder.knobKeys : [];
  const cur = String(sel.value || "").trim();
  const options = ['<option value="">(select knob)</option>'];
  for(const key of keys){
    const shown = escapeHtml(String(key || ""));
    options.push(`<option value="${shown}">${shown}</option>`);
  }
  sel.innerHTML = options.join("");
  if(cur && keys.includes(cur)) sel.value = cur;
}

function _builderRenderTemplatesList(){
  const sel = q("#rbTplSelect");
  if(!sel) return;
  const tpls = Array.isArray(Builder.templates) ? Builder.templates : [];
  const cur = String(sel.value||"");
  sel.innerHTML = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "(select template)";
  sel.appendChild(empty);
  for(const t of tpls){
    const opt = document.createElement("option");
    opt.value = String(t.name||"");
    opt.textContent = String(t.name||"");
    sel.appendChild(opt);
  }
  if(cur) sel.value = cur;
}

async function _builderLoadOnce(){
  if(Builder.loaded) return;
  Builder.loaded = true;
  try{
    Builder.cfg = await apiBuilderConfig();
    Builder.knobKeys = Array.isArray(Builder.cfg?.knob_keys) ? Builder.cfg.knob_keys : [];
  }catch(e){
    Builder.cfg = null;
    Builder.knobKeys = [];
    showToast("Builder config failed", e?.message||String(e));
  }
}

async function _builderRefreshLists(){
  await _builderEnsureRunsLoaded();
  try{
    const regs = await apiBuilderRegistries();
    Builder.registries = Array.isArray(regs?.registries) ? regs.registries : [];
  }catch(e){
    Builder.registries = [];
    showToast("Registry scan failed", e?.message||String(e));
  }
  try{
    const t = await apiBuilderTemplates();
    Builder.templates = Array.isArray(t?.templates) ? t.templates : [];
  }catch(_e){
    Builder.templates = [];
  }
}

function _builderInjectDefaults(){
  const cfg = Builder.cfg || {};
  const runsRootDefault = cfg?.builder?.runs_root_default || "experiments_data/emergence/runs";

  // Determinism toggles (PIS-7J2Q): persist via dedicated localStorage keys.
  // Default strict ON, legacy debug OFF.
  try{
    const strict = _rbGetLs(_RB_LS_STRICT_KEY, "1") === "1";
    const legacy = _rbGetLs(_RB_LS_LEGACY_KEY, "0") === "1";
    const planFreeze = _rbGetLs(_RB_LS_PLAN_FREEZE_KEY, "0") === "1";
    const cbStrict = q("#rbStrictDet");
    if(cbStrict) cbStrict.checked = !!strict;
    const cbLegacy = q("#rbLegacyDebugDet");
    if(cbLegacy) cbLegacy.checked = !!legacy;
    const cbPlanFreeze = q("#rbPlanFreeze");
    if(cbPlanFreeze) cbPlanFreeze.checked = !!planFreeze;
  }catch(_e){}

  // CLI builder default runs_dir
  const rd = q("#rbRunsDir");
  if(rd && !String(rd.value||"").trim()) rd.value = runsRootDefault;

  // Harness defaults
  const hr = q("#rbHarnessRunsRoot");
  if(hr && !String(hr.value||"").trim()) hr.value = runsRootDefault;
  const mn = q("#rbManifestOutRoot");
  if(mn && !String(mn.value||"").trim()){
    const fallbackOut = "./runtime/artifacts/manifests/generated";
    mn.value = String(cfg?.builder?.manifest_out_dir || fallbackOut);
  }
}

// -----------------------------
// Run Builder session persistence (localStorage)
// -----------------------------

const _BUILDER_LS_KEY = "eiel_run_builder_state_v1";

function _builderPersistSoon(){
  try{
    if(Builder._persistTimer) clearTimeout(Builder._persistTimer);
    Builder._persistTimer = setTimeout(_builderPersistNow, 200);
  }catch(_e){}
}

function _builderPersistNow(){
  try{
    const st = {
      schema: "run_builder_session_v1",
      utc: new Date().toISOString(),
      payload: {
        launch: {
          managed: true,
          active_job_id: String(Builder.activeLaunchJobId || "").trim(),
          lifecycle: String(Builder.launchLifecycle || "IDLE").trim(),
          plan_freeze: !!q("#rbPlanFreeze")?.checked,
          strict_determinism: !!q("#rbStrictDet")?.checked,
          legacy_debug_determinism: !!q("#rbLegacyDebugDet")?.checked,
        },
        cli: {
          entrypoint: q("#rbEntrypoint")?.value,
          runs_dir: q("#rbRunsDir")?.value,
          run_id: q("#rbRunId")?.value,
          run_tag: q("#rbRunTag")?.value,
          resume_from: q("#rbResume")?.value,
          resume_pick_run: q("#rbResumeRun")?.value,
          resume_pick_ckpt: q("#rbResumeCkpt")?.value,
          generations: q("#rbGenerations")?.value,
          steps: q("#rbSteps")?.value,
          seed: q("#rbSeed")?.value,
          pop: q("#rbPop")?.value,
          grid_size: q("#rbGrid")?.value,
          policy_tick_stride: q("#rbStride")?.value,
          checkpoint_period_gen: q("#rbCkptEvery")?.value,
          checkpoint_rotate: q("#rbCkptKeep")?.value,
          device: q("#rbDevice")?.value,
          raw_args: q("#rbRawArgs")?.value,
        },
        harness: {
          test_id: q("#rbHarnessName")?.value,
          runs_root: q("#rbHarnessRunsRoot")?.value,
          description: q("#rbHarnessDesc")?.value,
          intent_question: q("#rbHarnessQuestion")?.value,
          perf: !!q("#rbHarnessPerf")?.checked,
          enable_profiler: !!q("#rbHarnessProfiler")?.checked,
          seeds: q("#rbHarnessSeeds")?.value,
          generations: q("#rbHarnessGenerations")?.value,
          steps: q("#rbHarnessSteps")?.value,
          pop: q("#rbHarnessPop")?.value,
          grid_size: q("#rbHarnessGrid")?.value,
          stride: q("#rbHarnessStride")?.value,
          ckpt_every: q("#rbHarnessCkptEvery")?.value,
          ckpt_keep: q("#rbHarnessCkptKeep")?.value,
          expected_artifacts: q("#rbHarnessArtifacts")?.value,
          default_knobs: q("#rbHarnessDefaultKnobs")?.value,
          axes: _builderCollectAxesFromDom(),
          axis_flags: Builder.axisFlags,
        },
        manifest: {
          name: q("#rbManifestName")?.value,
          output_root: q("#rbManifestOutRoot")?.value,
          strict: !!q("#rbManifestStrict")?.checked,
          git_ref_override: q("#rbManifestGitRef")?.value,
          manual: q("#rbManifestManual")?.value,
        },
      },
    };
    localStorage.setItem(_BUILDER_LS_KEY, JSON.stringify(st));
  }catch(_e){}
}

function _builderRestoreFromStorage(){
  try{
    const raw = localStorage.getItem(_BUILDER_LS_KEY);
    if(!raw) return;
    const st = JSON.parse(raw);
    const p = st?.payload || {};

    // Launch settings
    // Managed mode is always-on in UC2.
    Builder.activeLaunchJobId = String(p?.launch?.active_job_id || "").trim();
    Builder.launchLifecycle = String(p?.launch?.lifecycle || "IDLE").trim().toUpperCase() || "IDLE";
    if(typeof p?.launch?.plan_freeze !== "undefined"){
      _rbSetLs(_RB_LS_PLAN_FREEZE_KEY, p.launch.plan_freeze ? "1" : "0");
      if(q("#rbPlanFreeze")) q("#rbPlanFreeze").checked = !!p.launch.plan_freeze;
    }
    if(typeof p?.launch?.strict_determinism !== "undefined"){
      _rbSetLs(_RB_LS_STRICT_KEY, p.launch.strict_determinism ? "1" : "0");
      if(q("#rbStrictDet")) q("#rbStrictDet").checked = !!p.launch.strict_determinism;
    }
    if(typeof p?.launch?.legacy_debug_determinism !== "undefined"){
      _rbSetLs(_RB_LS_LEGACY_KEY, p.launch.legacy_debug_determinism ? "1" : "0");
      if(q("#rbLegacyDebugDet")) q("#rbLegacyDebugDet").checked = !!p.launch.legacy_debug_determinism;
    }
    // Recover from stale persisted session state where lifecycle says RUNNING/STARTING
    // but there is no active job id to poll; otherwise Start remains locked forever.
    if(!Builder.activeLaunchJobId && (Builder.launchLifecycle === "RUNNING" || Builder.launchLifecycle === "STARTING")){
      Builder.launchLifecycle = "IDLE";
    }

    // CLI
    if(p.cli){
      if(q("#rbEntrypoint") && p.cli.entrypoint) q("#rbEntrypoint").value = p.cli.entrypoint;
      if(q("#rbRunsDir")) q("#rbRunsDir").value = p.cli.runs_dir || "";
      if(q("#rbRunId")) q("#rbRunId").value = p.cli.run_id || "";
      if(q("#rbRunTag")) q("#rbRunTag").value = p.cli.run_tag || "";
      if(q("#rbResume")) q("#rbResume").value = p.cli.resume_from || "";
      if(q("#rbGenerations")) q("#rbGenerations").value = p.cli.generations || "";
      if(q("#rbSteps")) q("#rbSteps").value = p.cli.steps || "";
      if(q("#rbSeed")) q("#rbSeed").value = p.cli.seed || "";
      if(q("#rbPop")) q("#rbPop").value = p.cli.pop || "";
      if(q("#rbGrid")) q("#rbGrid").value = p.cli.grid_size || "";
      if(q("#rbStride")) q("#rbStride").value = p.cli.policy_tick_stride || "";
      if(q("#rbCkptEvery")) q("#rbCkptEvery").value = p.cli.checkpoint_period_gen || "";
      if(q("#rbCkptKeep")) q("#rbCkptKeep").value = p.cli.checkpoint_rotate || "";
      if(q("#rbDevice")) q("#rbDevice").value = p.cli.device || "auto";
      if(q("#rbRawArgs")) q("#rbRawArgs").value = p.cli.raw_args || "";
    }

    // Harness
    if(p.harness){
      if(q("#rbHarnessName")) q("#rbHarnessName").value = p.harness.test_id || "";
      if(q("#rbHarnessRunsRoot")) q("#rbHarnessRunsRoot").value = p.harness.runs_root || "";
      if(q("#rbHarnessDesc")) q("#rbHarnessDesc").value = p.harness.description || "";
      if(q("#rbHarnessQuestion")) q("#rbHarnessQuestion").value = p.harness.intent_question || "";
      if(q("#rbHarnessPerf")) q("#rbHarnessPerf").checked = !!p.harness.perf;
      if(q("#rbHarnessProfiler")) q("#rbHarnessProfiler").checked = !!p.harness.enable_profiler;
      if(q("#rbHarnessSeeds")) q("#rbHarnessSeeds").value = p.harness.seeds || "";
      if(q("#rbHarnessGenerations")) q("#rbHarnessGenerations").value = p.harness.generations || "";
      if(q("#rbHarnessSteps")) q("#rbHarnessSteps").value = p.harness.steps || "";
      if(q("#rbHarnessPop")) q("#rbHarnessPop").value = p.harness.pop || "";
      if(q("#rbHarnessGrid")) q("#rbHarnessGrid").value = p.harness.grid_size || "";
      if(q("#rbHarnessStride")) q("#rbHarnessStride").value = p.harness.stride || "";
      if(q("#rbHarnessCkptEvery")) q("#rbHarnessCkptEvery").value = p.harness.ckpt_every || "";
      if(q("#rbHarnessCkptKeep")) q("#rbHarnessCkptKeep").value = p.harness.ckpt_keep || "";
      if(q("#rbHarnessArtifacts")) q("#rbHarnessArtifacts").value = p.harness.expected_artifacts || "";
      if(q("#rbHarnessDefaultKnobs")) q("#rbHarnessDefaultKnobs").value = p.harness.default_knobs || "";

      // Axes replace
      const axesBox = q("#rbAxes");
      if(axesBox){ axesBox.innerHTML = ""; }
      const axes = Array.isArray(p.harness.axes) ? p.harness.axes : [];
      if(axes.length){
        for(const ax of axes){
          _builderAddAxisRow(ax?.name||"axis", Array.isArray(ax?.values)?ax.values:[]);
        }
      }else{
        _builderAddAxisRow("case", []);
      }
      Builder.axisFlags = (p.harness.axis_flags && typeof p.harness.axis_flags === "object") ? p.harness.axis_flags : {};
      _builderRefreshCaseDropdown();
      _builderRenderFlagsForSelectedCase();
    }

    // Manifest
    if(p.manifest){
      if(q("#rbManifestName")) q("#rbManifestName").value = p.manifest.name || "";
      if(q("#rbManifestOutRoot")) q("#rbManifestOutRoot").value = p.manifest.output_root || "";
      if(q("#rbManifestStrict")) q("#rbManifestStrict").checked = !!p.manifest.strict;
      if(q("#rbManifestGitRef")) q("#rbManifestGitRef").value = p.manifest.git_ref_override || "";
      if(q("#rbManifestManual")) q("#rbManifestManual").value = p.manifest.manual || "";
    }
  }catch(_e){}
}

function _builderSanitizeComponent(s){
  const t0 = String(s||"").trim();
  // Match server-side sanitization semantics loosely.
  let t = t0.replace(/[^A-Za-z0-9._-]+/g, "_");
  t = t.replace(/^_+/, "").replace(/_+$/, "");
  return t || "my_harness";
}

function _builderRenderHarnessCmdPreview(){
  const out = q("#rbHarnessCmdOut");
  if(!out) return;
  const name = String(q("#rbHarnessName")?.value||"").trim();
  const base = _builderSanitizeComponent(name || "my_harness");
  const mod = `${_builderHarnessModulePrefix()}.${base}__harness`;
  const raw = `python -m ${mod}`;
  const prevRaw = String(Builder.lastHarnessCmdRaw || "");
  const prevKey = _builderHarnessReadinessKey(prevRaw);
  const nextKey = _builderHarnessReadinessKey(raw);
  Builder.lastHarnessCmdRaw = raw;
  // Only invalidate readiness when the effective harness command changed
  // (true edits). Routine re-renders should not force re-load/re-generate.
  if(prevRaw && prevKey && nextKey && prevKey !== nextKey){
    Builder.harnessLaunchReady = false;
  }
  _builderWriteCommandPreview(out, _builderApplyEnvPrefix(_builderCmdToManaged(raw)));
  _builderApplyStartButtonState();
}

function _builderAddAxisRow(name, values){
  const box = q("#rbAxes");
  if(!box) return;

  const row = document.createElement("div");
  row.className = "rb-axis-row";
  row.setAttribute("data-axis-name", String(name || "case"));
  row.innerHTML = `
    <div class="rb-axis-head">
      <div class="rb-axis-title">${escapeHtml(name || "case")}</div>
    </div>
    <textarea class="input rb-axis-values rb-w-axis-values" rows="6" placeholder="one value per line">${escapeHtml((values||[]).join("\n"))}</textarea>
  `;
  box.appendChild(row);

  safeOn(q(".rb-axis-values", row), "input", ()=>{
    _builderRefreshCaseDropdown();
    _builderRenderFlagsForSelectedCase();
    _builderRefreshHarnessDirtyState();
    _builderPersistSoon();
  });
}

function renderBuilderView(){
  const root = q("#view-builder");
  if(!root) return;

  // Keep builder state when switching tabs: don't nuke the DOM every time.
  if(Builder.domReady){
    _builderLoadOnce().then(async ()=>{
      await _builderRefreshLists();
      _builderRenderRegistryList();
      _builderRenderTemplatesList();
      _builderRenderEntrypoints();
      _builderRenderKnobKeyDatalist();
      _builderRenderResumePicker();
    });
    if(String(Builder.activeLaunchJobId || "").trim()){
      _builderScheduleLaunchPoll(true);
    }
    _builderRenderLaunchStatus();
    _builderRenderCliPreview();
    _builderRenderHarnessCmdPreview();
    _builderRefreshHarnessDirtyState();
    _builderRefreshRegistryActionState();
    if(!Builder._selfTestDone){
      Builder._selfTestDone = true;
      _builderRunSelfTests();
    }
    return;
  }
  Builder.domReady = true;

  // Skeleton
  root.innerHTML = `
    <div class="card rb-action-pane">
      <div class="rb-action-head">
        <div>
          <div class="rb-title">Harness + Registry</div>
          <div class="muted">UC2 workflow</div>
        </div>
      </div>

      <div class="rb-toolbar-field rb-toolbar-field--registry">
        <label class="rb-label" for="rbRegistryLoadSelect">Registry</label>
        <select id="rbRegistryLoadSelect" class="input rb-w-registry"></select>
      </div>

      <div class="rb-toolbar rb-action-controls">
        <div class="rb-toggle-group">
          <label class="control-inline rb-toggle" title="Add --perf to every generated case. Loaded from the selected registry when available.">
            Perf
            <input id="rbHarnessPerf" type="checkbox"/>
          </label>
          <label class="control-inline rb-toggle" title="Add --enable_profiler to every generated case. Loaded from the selected registry when available.">
            Profiler
            <input id="rbHarnessProfiler" type="checkbox"/>
          </label>
        </div>
        <button id="rbHarnessGen" class="btn ok small" type="button">Generate</button>
        <button id="rbHarnessCmdCopy" class="btn small" type="button">Copy command</button>
        <button id="rbHarnessCmdStart" class="btn small" type="button">Start</button>
        <button id="rbRegistryDeleteBtn" class="btn danger small" type="button">Delete registries</button>
      </div>

      <div class="rb-toolbar rb-toolbar-meta">
        <span id="rbRegistryLoadStatus" class="muted"></span>
        <span id="rbHarnessDirtyState" class="muted"></span>
        <span id="rbHarnessStatus" class="muted"></span>
      </div>

      <div class="rb-command-stack">
        <div class="rb-label">Command</div>
        <div id="rbHarnessCmdOut" class="rb-command-label mono"></div>
      </div>

      <details class="rb-utilities">
        <summary>Builder utilities</summary>
        <div class="rb-utilities-grid">
          <div class="row rb-utilities-row">
            <span class="muted" title="Builder launches are always managed (durable job + report_dir + stop sentinel).">Managed launch: always on</span>
            <label class="control-inline" title="Adds PYTHONHASHSEED=0 and EIEL_STRICT_DETERMINISM=1. Required for strict determinism and replay.">
              Deterministic
              <input id="rbStrictDet" type="checkbox" checked/>
            </label>
            <label class="control-inline" title="Pre-run plane governance. When enabled, the launched run sets EIEL_ENFORCE_PLAN_FREEZE=1.">
              Plan Freeze
              <input id="rbPlanFreeze" type="checkbox"/>
            </label>
            <label class="control-inline" title="Legacy compatibility. Prefer EIEL_STRICT_DETERMINISM.">
              DEBUG_DETERMINISM
              <input id="rbLegacyDebugDet" type="checkbox"/>
            </label>
          </div>
          <div class="row rb-utilities-row">
            <select id="rbTplSelect" class="input small rb-w-template" title="Load a saved template"></select>
            <button id="rbTplLoad" class="btn small" type="button">Load</button>
            <input id="rbTplName" class="input small rb-w-template" placeholder="template name"/>
            <button id="rbTplSave" class="btn small" type="button">Save</button>
          </div>
        </div>
      </details>
    </div>

    <div class="builder-grid builder-grid--builder">
      <div class="card rb-pane rb-pane--config">
        <details class="rb-pane-shell">
          <summary class="rb-pane-summary">Simulation Config</summary>
          <div class="rb-form">
          <div class="rb-three">
            <div>
              <label class="rb-label">Name</label>
              <input id="rbHarnessName" class="input rb-w-name" placeholder="my_experiment_pack"/>
            </div>
            <div>
              <label class="rb-label">Runs root</label>
              <input id="rbHarnessRunsRoot" class="input rb-w-path" placeholder="experiments_data/emergence/runs"/>
            </div>
            <div>
              <label class="rb-label">Description</label>
              <input id="rbHarnessDesc" class="input rb-w-note" placeholder="short note"/>
            </div>
          </div>

          <div>
            <div>
              <label class="rb-label">Question</label>
              <textarea id="rbHarnessQuestion" class="input rb-textarea-compact rb-w-note" rows="2" placeholder="what are we trying to learn?"></textarea>
            </div>
          </div>

          <div class="rb-grid rb-grid--compact rb-grid--numeric">
            <div>
              <label class="rb-label">Seeds</label>
              <input id="rbHarnessSeeds" class="input rb-w-seeds" value="1,2,3"/>
            </div>
            <div>
              <label class="rb-label">Generations</label>
              <input id="rbHarnessGenerations" class="input rb-w-num" type="number" min="1" step="1" value="12"/>
            </div>
            <div>
              <label class="rb-label">Steps</label>
              <input id="rbHarnessSteps" class="input rb-w-num" type="number" min="1" step="1" value="150"/>
            </div>
            <div>
              <label class="rb-label">Pop</label>
              <input id="rbHarnessPop" class="input rb-w-num" type="number" min="1" step="1" value="128"/>
            </div>
            <div>
              <label class="rb-label">Grid</label>
              <input id="rbHarnessGrid" class="input rb-w-num" type="number" min="1" step="1" placeholder=""/>
            </div>
            <div>
              <label class="rb-label">Stride</label>
              <input id="rbHarnessStride" class="input rb-w-num" type="number" min="1" step="1" value="2"/>
            </div>
            <div>
              <label class="rb-label">Ckpt every</label>
              <input id="rbHarnessCkptEvery" class="input rb-w-num" type="number" min="0" step="1" value="0"/>
            </div>
            <div>
              <label class="rb-label">Ckpt keep</label>
              <input id="rbHarnessCkptKeep" class="input rb-w-num" type="number" min="0" step="1" value="0"/>
            </div>
          </div>

          <div class="rb-subtitle">Shared knobs</div>
          <textarea id="rbHarnessDefaultKnobs" class="input rb-textarea-short rb-w-textarea" rows="3" placeholder="sim_cfg.late_pred_cost_weight=4.0&#10;sim_cfg.some_toggle=true"></textarea>

          <div class="rb-subtitle">Baselines and variants</div>
          <div id="rbAxes" class="rb-axes"></div>

          <div class="rb-subtitle">Case flags</div>
          <div class="rb-case-grid">
            <div>
              <label class="rb-label">Case</label>
              <select id="rbCaseSelect" class="input rb-w-select"></select>
            </div>
            <div>
              <label class="rb-label">Knob</label>
              <div class="rb-case-inline">
                <select id="rbKnobKey" class="input rb-w-note"></select>
                <input id="rbKnobVal" class="input rb-w-num" placeholder="4.0"/>
                <button id="rbKnobAdd" class="btn small" type="button">Add</button>
              </div>
            </div>
            <div>
              <label class="rb-label">Raw flag</label>
              <div class="rb-case-inline rb-case-inline--flag">
                <input id="rbRawFlag" class="input rb-w-flag" placeholder="--some_flag"/>
                <button id="rbFlagAdd" class="btn small" type="button">Add</button>
              </div>
            </div>
          </div>
          <div class="rb-case-tokens">
            <div>
              <div class="row rb-inline-actions rb-inline-actions--spread">
                <label class="rb-label">Selected case tokens</label>
                <button id="rbCaseFlagsCopy" class="btn small" type="button">Copy</button>
              </div>
              <textarea id="rbCaseFlagsOut" class="input mono rb-textarea-tall rb-w-textarea rb-w-case-flags" rows="6" wrap="off" placeholder="--knob&#10;sim_cfg.foo=1"></textarea>
            </div>
          </div>
          <div id="rbHarnessResult" class="rb-result"></div>
          </div>
        </details>
      </div>

      <div class="card rb-pane rb-pane--manifest">
        <details class="rb-pane-shell">
          <summary class="rb-pane-summary">Manifest Generation</summary>
          <div class="rb-form">
          <div class="rb-two">
            <div>
              <label class="rb-label">Manifest name</label>
              <input id="rbManifestName" class="input rb-w-name" placeholder="my_manifest"/>
            </div>
            <div>
              <label class="rb-label">Output root</label>
              <input id="rbManifestOutRoot" class="input rb-w-path" placeholder="./runtime/artifacts/manifests/generated/my_manifest"/>
            </div>
          </div>

          <div style="margin-top:8px;">
            <label class="rb-label">git_ref override</label>
            <input id="rbManifestGitRef" class="input mono rb-w-gitref" placeholder="40-char SHA (useful if snapshot has no .git)"/>
          </div>

          <label class="rb-reg-head">Registries</label>
          <div id="rbRegistries" class="rb-reg-list"></div>
          <div class="row rb-inline-actions" style="margin-top:8px;">
            <button id="rbRegsRefresh" class="btn small" type="button">Refresh registry list</button>
            <label class="control-inline"><input id="rbManifestStrict" type="checkbox"/> Strict verify</label>
          </div>

          <label class="rb-label" style="margin-top:10px;">Manual registry paths</label>
          <textarea id="rbManifestManual" class="input rb-textarea-compact rb-w-textarea" rows="2" placeholder="experiments/registry/v1/generated/foo__harness.yaml"></textarea>

          <div class="row rb-inline-actions" style="margin-top:10px;">
            <button id="rbManifestGen" class="btn ok" type="button">Generate manifest</button>
            <span id="rbManifestStatus" class="muted"></span>
          </div>


          <div style="margin-top:12px;">
            <label class="rb-label">Manifest start command</label>
            <textarea id="rbManifestCmdOut" class="input mono rb-command-out rb-w-command" rows="2" readonly></textarea>
            <div class="row rb-inline-actions" style="margin-top:6px;">
              <button id="rbManifestCmdCopy" class="btn small" type="button">Copy</button>
              <button id="rbManifestCmdStart" class="btn small" type="button">Start</button>
            </div>
          </div>

          <div id="rbManifestResult" class="rb-result"></div>
          </div>
        </details>
      </div>

    </div>
  `;
  // Render System Runs table (unified active runs).
  try{
    const box = q("#systemRunsTable", root);
    const data = State.systemRuns || {};
    const active = Array.isArray(data.active)
      ? data.active
      : ([]).concat(Array.isArray(data.managed) ? data.managed : [], Array.isArray(data.unmanaged) ? data.unmanaged : []);
    if(box){
      const rows = [];
      for(const r of active){
        const rk = escapeHtml(String(r.run_key||""));
        const shown = escapeHtml(displayId(r, {fallback: String(r.run_key || "")}));
        const st = escapeHtml(String(r.state||""));
        const pid = r.pid ? escapeHtml(String(r.pid)) : "";
        const gen = (r.gen===null||r.gen===undefined) ? "" : escapeHtml(String(r.gen));
        const age = (r.heartbeat_age_s===null||r.heartbeat_age_s===undefined) ? "n/a" : escapeHtml(String(r.heartbeat_age_s));
        const source = String(r.source || r.origin || "").trim().toLowerCase();
        const rkRaw = String(r.run_key || "");
        const actionBtn = (source === "external")
          ? `<button class="btn small danger js-stop-external" data-run-key="${escapeHtml(rkRaw)}" title="Stop external run (SIGTERM)">Stop</button>`
          : "";
        rows.push(`<tr><td><a href="#run/${encodeURIComponent(rk)}" title="${rk}">${shown}</a></td><td><span class="badge ${st==="HUNG"?"b-fail":(st==="RUNNING"?"b-pass":"b-unknown")}">${st}</span></td><td class="mono">${pid}</td><td class="mono">${gen}</td><td class="mono">${age}s</td><td>${actionBtn}</td></tr>`);
      }
      if(!rows.length){
        box.innerHTML = `<div class="muted">No runs detected.</div>`;
      }else{
        box.innerHTML = `<table class="tbl"><thead><tr><th>Run</th><th>State</th><th>PID</th><th>Gen</th><th>Heartbeat</th><th>Action</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
        qa(".js-stop-external", box).forEach((btn)=>{
          safeOn(btn, "click", async ()=>{
            const rkRaw = String(btn.getAttribute("data-run-key") || "").trim();
            if(!rkRaw) return;
            const ok = window.confirm(`Stop external run "${rkRaw}" with SIGTERM?`);
            if(!ok) return;
            try{
              await fetchJson(`/api/run/${encodeURIComponent(rkRaw)}/stop`, { method: "POST" });
              showToast("Stop requested", `SIGTERM sent for ${rkRaw}`);
              await _refreshSystemRuns();
              if(State.view === "overview") renderOverviewView();
            }catch(e){
              showToast("Stop failed", String(e?.message || e));
            }
          });
        });
      }
    }
  }catch(_e){}


  // Wiring (best-effort, never crash)
  _builderLoadOnce().then(async ()=>{
    await _builderRefreshLists();
    _builderRenderRegistryList();
    _builderRenderTemplatesList();
    _builderRenderEntrypoints();
    _builderRenderKnobKeyDatalist();
    _builderRenderResumePicker();

    // Restore last session (localStorage) before injecting defaults.
    _builderRestoreFromStorage();
    // Add default axis row on first render
    const axesBox = q("#rbAxes");
    if(axesBox && !qa(".rb-axis-row", axesBox).length){
      _builderAddAxisRow("case", []);
    }
    _builderRefreshCaseDropdown();
    _builderInjectDefaults();
    _builderRenderCliPreview();
    _builderRenderHarnessCmdPreview();
    _builderRefreshHarnessDirtyState();
    _builderRefreshRegistryActionState();
    _builderRenderLaunchStatus();
    if(String(Builder.activeLaunchJobId || "").trim()){
      Builder.launchPollCount = 0;
      // Do not hard-disable Start when merely restoring/polling an existing job.
      // Operators should still be able to launch a new run.
      _builderSetStartButtonsDisabled(false);
      _builderScheduleLaunchPoll(true);
    }else{
      _builderSetStartButtonsDisabled(false);
    }
    if(!Builder._selfTestDone){
      Builder._selfTestDone = true;
      _builderRunSelfTests();
    }
  });

  // CLI builder events
  safeOn(q("#rbCmdGen"), "click", _builderRenderCliPreview);
  safeOn(q("#rbResumeRun"), "change", async ()=>{
    await _builderRefreshResumeCkpts();
    _builderPersistNow();
  });
  safeOn(q("#rbResumeCkpt"), "change", _builderPersistNow);
  safeOn(q("#rbResumeUse"), "click", ()=>{
    const rk = String(q("#rbResumeRun")?.value||"").trim();
    const rel = String(q("#rbResumeCkpt")?.value||"").trim();
    if(!rk || !rel){
      showToast("Resume picker", "Select a run and a checkpoint");
      return;
    }
    const out = q("#rbResume");
    if(out) out.value = `${rk}/${rel}`;
    _builderPersistNow();
    _builderRenderCliPreview();
  });
  safeOn(q("#rbCmdCopy"), "click", async ()=>{
    const cmd = String(_builderApplyEnvPrefix(_builderCmdToManaged(Builder.lastCliCmdRaw || "")) || "").trim();
    if(!cmd) return;
    const ok = await copyTextToClipboard(cmd);
    showToast("Copied", ok ? "CLI command copied to clipboard" : "Copy failed");
  });

  safeOn(q("#rbCmdStart"), "click", async ()=>{
    const btn = q("#rbCmdStart");
    const cmd = String(_builderApplyEnvPrefix(_builderCmdToManaged(Builder.lastCliCmdRaw || "")) || "").trim();
    if(!cmd){
      showToast("Nothing to start", "Generate a command first");
      return;
    }
    await _builderStartManagedCommand(cmd, "Run", btn);
  });

  // Determinism toggles (PIS-7J2Q)
  safeOn(q("#rbStrictDet"), "change", ()=>{
    _rbSetLs(_RB_LS_STRICT_KEY, q("#rbStrictDet")?.checked ? "1" : "0");
    _builderRenderCliPreview();
    _builderRenderHarnessCmdPreview();
    _builderRefreshHarnessDirtyState();
    try{
      const m = q("#rbManifestCmdOut");
      if(m) _builderWriteCommandPreview(m, _builderApplyEnvPrefix(_builderCmdToManaged(Builder.lastManifestCmdRaw || String(_builderReadTextLike(m) || ""))));
    }catch(_e){}
    _builderPersistSoon();
  });
  safeOn(q("#rbLegacyDebugDet"), "change", ()=>{
    _rbSetLs(_RB_LS_LEGACY_KEY, q("#rbLegacyDebugDet")?.checked ? "1" : "0");
    _builderRenderCliPreview();
    _builderRenderHarnessCmdPreview();
    _builderRefreshHarnessDirtyState();
    try{
      const m = q("#rbManifestCmdOut");
      if(m) _builderWriteCommandPreview(m, _builderApplyEnvPrefix(_builderCmdToManaged(Builder.lastManifestCmdRaw || String(_builderReadTextLike(m) || ""))));
    }catch(_e){}
    _builderPersistSoon();
  });
  safeOn(q("#rbPlanFreeze"), "change", ()=>{
    _rbSetLs(_RB_LS_PLAN_FREEZE_KEY, q("#rbPlanFreeze")?.checked ? "1" : "0");
    _builderRenderCliPreview();
    _builderRenderHarnessCmdPreview();
    _builderRefreshHarnessDirtyState();
    try{
      const m = q("#rbManifestCmdOut");
      if(m) _builderWriteCommandPreview(m, _builderApplyEnvPrefix(_builderCmdToManaged(Builder.lastManifestCmdRaw || String(_builderReadTextLike(m) || ""))));
    }catch(_e){}
    _builderPersistSoon();
  });
  ["#rbEntrypoint", "#rbRunsDir", "#rbRunId", "#rbRunTag", "#rbResume", "#rbGenerations", "#rbSteps", "#rbSeed", "#rbPop", "#rbGrid", "#rbStride", "#rbCkptEvery", "#rbCkptKeep", "#rbDevice", "#rbRawArgs"].forEach(sel=>{
    safeOn(q(sel), "input", ()=>{ _builderRenderCliPreview(); _builderPersistSoon(); });
    safeOn(q(sel), "change", ()=>{ _builderRenderCliPreview(); _builderPersistSoon(); });
  });

  // Axes
  safeOn(q("#rbAxisAdd"), "click", ()=>{ _builderAddAxisRow("axis", []); _builderRefreshCaseDropdown(); _builderRefreshHarnessDirtyState(); _builderPersistSoon(); });

  // Case flags
  safeOn(q("#rbCaseSelect"), "change", ()=>{ _builderRenderFlagsForSelectedCase(); });
  safeOn(q("#rbCaseFlagsOut"), "input", ()=>{ _builderUpdateFlagsFromEditor(); _builderRefreshHarnessDirtyState(); });
  safeOn(q("#rbCaseFlagsCopy"), "click", async ()=>{
    const out = q("#rbCaseFlagsOut");
    if(!out) return;
    const ok = await copyTextToClipboard(out.value);
    showToast("Copied", ok ? "case flags copied" : "copy failed");
  });
  safeOn(q("#rbKnobAdd"), "click", ()=>{ _builderAddKnobToCase(); });
  safeOn(q("#rbFlagAdd"), "click", ()=>{ _builderAddRawFlagToCase(); });

  // Harness edit persistence
  [
    "#rbHarnessName", "#rbHarnessRunsRoot", "#rbHarnessDesc", "#rbHarnessQuestion",
    "#rbHarnessSeeds", "#rbHarnessGenerations", "#rbHarnessSteps", "#rbHarnessPop",
    "#rbHarnessGrid", "#rbHarnessStride", "#rbHarnessCkptEvery", "#rbHarnessCkptKeep",
    "#rbHarnessArtifacts", "#rbHarnessDefaultKnobs", "#rbHarnessPerf", "#rbHarnessProfiler",
  ].forEach(sel=>{
    safeOn(q(sel), "input", ()=>{ if(sel==="#rbHarnessName") _builderRenderHarnessCmdPreview(); _builderRefreshHarnessDirtyState(); _builderPersistSoon(); });
    safeOn(q(sel), "change", ()=>{ if(sel==="#rbHarnessName") _builderRenderHarnessCmdPreview(); _builderRefreshHarnessDirtyState(); _builderPersistSoon(); });
  });

  // Harness start command copy
  safeOn(q("#rbHarnessCmdCopy"), "click", async ()=>{
    const cmd = String(_builderApplyEnvPrefix(_builderCmdToManaged(Builder.lastHarnessCmdRaw || "")) || "").trim();
    if(!cmd) return;
    const ok = await copyTextToClipboard(cmd);
    showToast("Copied", ok ? "harness start command copied" : "copy failed");
  });

  safeOn(q("#rbHarnessCmdStart"), "click", async ()=>{
    const btn = q("#rbHarnessCmdStart");
    const cmd = String(_builderApplyEnvPrefix(_builderCmdToManaged(Builder.lastHarnessCmdRaw || "")) || "").trim();
    if(!cmd){
      showToast("Nothing to start", "Generate a harness first");
      return;
    }
    await _builderStartManagedCommand(cmd, "Harness", btn);
  });

  safeOn(q("#rbRegistryLoadSelect"), "change", async (ev)=>{
    const path = String(ev?.target?.value || "").trim();
    _builderSetRegistrySelection(path);
    if(!path){
      _builderRefreshRegistryActionState();
      return;
    }
    await _builderLoadRegistrySelectionWithGuard(path);
  });
  safeOn(q("#rbRegistryDeleteBtn"), "click", async ()=>{
    try{
      _builderOpenDeleteRegistriesModal();
    }catch(e){
      showToast("Delete failed", e?.message||String(e));
    }
  });

  // Harness generate
  safeOn(q("#rbHarnessGen"), "click", async ()=>{
    const status = q("#rbHarnessStatus");
    const resBox = q("#rbHarnessResult");
    if(status) status.textContent = "working...";
    if(resBox) resBox.innerHTML = "";

    let payload;
    try{ payload = _builderBuildHarnessPayload(); }
    catch(e){ showToast("Invalid harness", e?.message||String(e)); if(status) status.textContent = ""; return; }

    try{
      const res = await apiBuilderGenerateHarness(payload);
      if(status) status.textContent = "done";

      const ok = !!res?.ok;
      const issues = Array.isArray(res?.issues) ? res.issues : [];
      const zipUrl = res?.download_url ? String(res.download_url) : "";
      const shown = displayId(res, {fallback: String(res?.name || "")});

      const lines = [];
      lines.push(`<div class="row">${badge(ok?"OK":"ISSUES", ok?"b-pass":"b-fail")} <span class="muted">${escapeHtml(shown)}</span></div>`);
      lines.push(`<div class="muted">registry: <code>${escapeHtml(res?.registry_path||"")}</code></div>`);
      lines.push(`<div class="muted">harness: <code>${escapeHtml(res?.harness_path||"")}</code></div>`);
      if(res?.run_command){
        // Prefer server-provided managed command, but still enforce wrapper_root
        // and env prefixes at the UI layer.
        const shownServer = String(res.run_command_managed || res.run_command || "");
        const rawServer = String(res.run_command || shownServer || "");
        Builder.lastHarnessCmdRaw = rawServer;

        const shownManaged = _builderApplyEnvPrefix(_builderCmdToManaged(shownServer));
        lines.push(`<div class="muted">start: <code>${escapeHtml(shownManaged)}</code></div>`);
        const cmdOut = q("#rbHarnessCmdOut");
        _builderWriteCommandPreview(cmdOut, shownManaged);
        Builder.harnessLaunchReady = true;
      }else{
        _builderRenderHarnessCmdPreview();
      }
      if(zipUrl){
        lines.push(`<div style="margin-top:6px;"><a class="link" href="${escapeHtml(zipUrl)}">Download zip</a></div>`);
      }
      if(issues.length){
        lines.push(`<div class="rb-issues">${issues.map(x=>`<div>• ${escapeHtml(x)}</div>`).join("")}</div>`);
      }
      if(res?.registry_preview){
        lines.push(`<details style="margin-top:8px;"><summary>registry preview</summary><pre class="rb-pre">${escapeHtml(res.registry_preview)}</pre></details>`);
      }
      if(res?.harness_preview){
        lines.push(`<details style="margin-top:8px;"><summary>harness preview</summary><pre class="rb-pre">${escapeHtml(res.harness_preview)}</pre></details>`);
      }
      if(resBox) resBox.innerHTML = lines.join("");

      // Refresh registry list so UC3 sees it.
      Builder.loadedHarnessExists = true;
      Builder.loadedHarnessPath = String(res?.harness_path || "").trim();
      if(res?.registry_path){
        _builderSetRegistrySelection(String(res.registry_path || ""));
        _builderSetHarnessBaseline(String(res.registry_path || ""), String(res?.harness_path || ""), true);
      }else{
        Builder.harnessBaselineSignature = _builderCurrentHarnessSignature();
        Builder.loadedHarnessExists = true;
      }
      await _builderRefreshLists();
      _builderRenderRegistryList();
      _builderRefreshHarnessDirtyState();
    }catch(e){
      if(status) status.textContent = "failed";
      showToast("Harness generation failed", e?.message||String(e));
    }
  });

  // Manifest list refresh
  safeOn(q("#rbRegsRefresh"), "click", async ()=>{
    await _builderRefreshLists();
    _builderRenderRegistryList();
    showToast("Refreshed", "registry list updated");
  });

  ["#rbManifestName", "#rbManifestOutRoot", "#rbManifestGitRef", "#rbManifestStrict", "#rbManifestManual"].forEach(sel=>{
    safeOn(q(sel), "input", _builderPersistSoon);
    safeOn(q(sel), "change", _builderPersistSoon);
  });

  // Manifest generate
  safeOn(q("#rbManifestGen"), "click", async ()=>{
    const status = q("#rbManifestStatus");
    const resBox = q("#rbManifestResult");
    if(status) status.textContent = "working...";
    if(resBox) resBox.innerHTML = "";

    let payload;
    try{ payload = _builderBuildManifestPayload(); }
    catch(e){ showToast("Invalid manifest", e?.message||String(e)); if(status) status.textContent = ""; return; }

    if(!payload.registry_paths || !payload.registry_paths.length){
      showToast("No registries", "Select at least 1 registry (checkbox or manual path)");
      if(status) status.textContent = "";
      return;
    }

    try{
      const res = await apiBuilderGenerateManifest(payload);
      if(status) status.textContent = "done";
      // Populate manifest start command preview (best-effort).
      try{
        const cmdOut = q("#rbManifestCmdOut");
        const raw = String(res?.recommended_start_cmd || "").trim() || (res?.manifest_path ? (`python -m ${_builderManifestRunnerModule()} --manifest ${String(res.manifest_path)}`) : "");
        Builder.lastManifestCmdRaw = raw;
        const shown = _builderApplyEnvPrefix(_builderCmdToManaged(raw));
        _builderWriteCommandPreview(cmdOut, shown);
        _builderApplyStartButtonState();
      }catch(_e){}

      const ok = !!res?.ok;
      const issues = Array.isArray(res?.issues) ? res.issues : [];
      const zipUrl = res?.download_url ? String(res.download_url) : "";
      const lines = [];
      lines.push(`<div class="row">${badge(ok?"OK":"ISSUES", ok?"b-pass":"b-fail")} <span class="muted">jobs=${escapeHtml(String(res?.job_count ?? ""))}</span></div>`);
      lines.push(`<div class="muted">manifest: <code>${escapeHtml(res?.manifest_path||"")}</code></div>`);
      lines.push(`<div class="muted">git_ref: <code>${escapeHtml(res?.git_ref||"")}</code></div>`);
      if(zipUrl){
        lines.push(`<div style="margin-top:6px;"><a class="link" href="${escapeHtml(zipUrl)}">Download zip</a></div>`);
      }
      if(issues.length){
        lines.push(`<div class="rb-issues">${issues.map(x=>`<div>• ${escapeHtml(x)}</div>`).join("")}</div>`);
      }
      if(res?.manifest_preview){
        lines.push(`<details style="margin-top:8px;" open><summary>manifest preview</summary><pre class="rb-pre">${escapeHtml(res.manifest_preview)}</pre></details>`);
      }
      if(resBox) resBox.innerHTML = lines.join("");
    }catch(e){
      if(status) status.textContent = "failed";
      showToast("Manifest generation failed", e?.message||String(e));
    }
  });

  // Manifest command copy/start
  safeOn(q("#rbManifestCmdCopy"), "click", async ()=>{
    const cmd = String(_builderApplyEnvPrefix(_builderCmdToManaged(Builder.lastManifestCmdRaw || "")) || "").trim();
    if(!cmd) return;
    const ok = await copyTextToClipboard(cmd);
    showToast("Copied", ok ? "manifest start command copied" : "copy failed");
  });

  safeOn(q("#rbManifestCmdStart"), "click", async ()=>{
    const btn = q("#rbManifestCmdStart");
    const cmd = String(_builderApplyEnvPrefix(_builderCmdToManaged(Builder.lastManifestCmdRaw || "")) || "").trim();
    if(!cmd){
      showToast("Nothing to start", "Generate a manifest first");
      return;
    }
    await _builderStartManagedCommand(cmd, "Manifest", btn);
  });


  // Templates
  safeOn(q("#rbTplSave"), "click", async ()=>{
    const name = String(q("#rbTplName")?.value||"").trim();
    if(!name){ showToast("Template name missing", "enter a name"); return; }
    const obj = {
      name: name,
      schema_version: "run_builder_template_v1",
      utc: new Date().toISOString(),
      payload: {
        launch: {
          plan_freeze: !!q("#rbPlanFreeze")?.checked,
          strict_determinism: !!q("#rbStrictDet")?.checked,
          legacy_debug_determinism: !!q("#rbLegacyDebugDet")?.checked,
        },
        cli: {
          entrypoint: q("#rbEntrypoint")?.value,
          runs_dir: q("#rbRunsDir")?.value,
          run_id: q("#rbRunId")?.value,
          run_tag: q("#rbRunTag")?.value,
          resume_from: q("#rbResume")?.value,
          resume_pick_run: q("#rbResumeRun")?.value,
          resume_pick_ckpt: q("#rbResumeCkpt")?.value,
          generations: q("#rbGenerations")?.value,
          steps: q("#rbSteps")?.value,
          seed: q("#rbSeed")?.value,
          pop: q("#rbPop")?.value,
          grid_size: q("#rbGrid")?.value,
          policy_tick_stride: q("#rbStride")?.value,
          checkpoint_period_gen: q("#rbCkptEvery")?.value,
          checkpoint_rotate: q("#rbCkptKeep")?.value,
          device: q("#rbDevice")?.value,
          raw_args: q("#rbRawArgs")?.value,
        },
        harness: {
          test_id: q("#rbHarnessName")?.value,
          runs_root: q("#rbHarnessRunsRoot")?.value,
          description: q("#rbHarnessDesc")?.value,
          intent_question: q("#rbHarnessQuestion")?.value,
          perf: !!q("#rbHarnessPerf")?.checked,
          enable_profiler: !!q("#rbHarnessProfiler")?.checked,
          seeds: q("#rbHarnessSeeds")?.value,
          generations: q("#rbHarnessGenerations")?.value,
          steps: q("#rbHarnessSteps")?.value,
          pop: q("#rbHarnessPop")?.value,
          grid_size: q("#rbHarnessGrid")?.value,
          stride: q("#rbHarnessStride")?.value,
          ckpt_every: q("#rbHarnessCkptEvery")?.value,
          ckpt_keep: q("#rbHarnessCkptKeep")?.value,
          expected_artifacts: q("#rbHarnessArtifacts")?.value,
          default_knobs: q("#rbHarnessDefaultKnobs")?.value,
          axes: _builderCollectAxesFromDom(),
          axis_flags: Builder.axisFlags,
        },
        manifest: {
          name: q("#rbManifestName")?.value,
          output_root: q("#rbManifestOutRoot")?.value,
          strict: !!q("#rbManifestStrict")?.checked,
          git_ref_override: q("#rbManifestGitRef")?.value,
          manual: q("#rbManifestManual")?.value,
        },
      },
    };
    try{
      await apiBuilderSaveTemplate(obj);
      showToast("Template saved", name);
      await _builderRefreshLists();
      _builderRenderTemplatesList();
    }catch(e){
      showToast("Save failed", e?.message||String(e));
    }
  });

  safeOn(q("#rbTplLoad"), "click", async ()=>{
    const name = String(q("#rbTplSelect")?.value||"").trim();
    if(!name){ showToast("No template selected", "pick one"); return; }
    try{
      const res = await apiBuilderLoadTemplate({name});
      const p = res?.payload || {};
      if(p.launch){
        if(q("#rbPlanFreeze")) q("#rbPlanFreeze").checked = !!p.launch.plan_freeze;
        if(q("#rbStrictDet")) q("#rbStrictDet").checked = !!p.launch.strict_determinism;
        if(q("#rbLegacyDebugDet")) q("#rbLegacyDebugDet").checked = !!p.launch.legacy_debug_determinism;
        _rbSetLs(_RB_LS_PLAN_FREEZE_KEY, q("#rbPlanFreeze")?.checked ? "1" : "0");
        _rbSetLs(_RB_LS_STRICT_KEY, q("#rbStrictDet")?.checked ? "1" : "0");
        _rbSetLs(_RB_LS_LEGACY_KEY, q("#rbLegacyDebugDet")?.checked ? "1" : "0");
      }
      // CLI
      if(p.cli){
        if(q("#rbEntrypoint")) q("#rbEntrypoint").value = p.cli.entrypoint || "engine.main";
        if(q("#rbRunsDir")) q("#rbRunsDir").value = p.cli.runs_dir || "";
        if(q("#rbRunId")) q("#rbRunId").value = p.cli.run_id || "";
        if(q("#rbRunTag")) q("#rbRunTag").value = p.cli.run_tag || "";
        if(q("#rbResume")) q("#rbResume").value = p.cli.resume_from || "";
        if(q("#rbResumeRun")) q("#rbResumeRun").value = p.cli.resume_pick_run || "";
        // ckpt list is populated async; stash desired selection
        Builder._resumePickWanted = {run: (p.cli.resume_pick_run||""), ckpt: (p.cli.resume_pick_ckpt||"")};
        if(q("#rbGenerations")) q("#rbGenerations").value = p.cli.generations || "";
        if(q("#rbSteps")) q("#rbSteps").value = p.cli.steps || "";
        if(q("#rbSeed")) q("#rbSeed").value = p.cli.seed || "";
        if(q("#rbPop")) q("#rbPop").value = p.cli.pop || "";
        if(q("#rbGrid")) q("#rbGrid").value = p.cli.grid_size || "";
        if(q("#rbStride")) q("#rbStride").value = p.cli.policy_tick_stride || "";
        if(q("#rbCkptEvery")) q("#rbCkptEvery").value = p.cli.checkpoint_period_gen || "";
        if(q("#rbCkptKeep")) q("#rbCkptKeep").value = p.cli.checkpoint_rotate || "";
        if(q("#rbDevice")) q("#rbDevice").value = p.cli.device || "auto";
        if(q("#rbRawArgs")) q("#rbRawArgs").value = p.cli.raw_args || "";
      }
      // Harness
      if(p.harness){
        if(q("#rbHarnessName")) q("#rbHarnessName").value = p.harness.test_id || "";
        if(q("#rbHarnessRunsRoot")) q("#rbHarnessRunsRoot").value = p.harness.runs_root || "";
        if(q("#rbHarnessDesc")) q("#rbHarnessDesc").value = p.harness.description || "";
        if(q("#rbHarnessQuestion")) q("#rbHarnessQuestion").value = p.harness.intent_question || "";
        if(q("#rbHarnessPerf")) q("#rbHarnessPerf").checked = !!p.harness.perf;
        if(q("#rbHarnessProfiler")) q("#rbHarnessProfiler").checked = !!p.harness.enable_profiler;
        if(q("#rbHarnessSeeds")) q("#rbHarnessSeeds").value = p.harness.seeds || "";
        if(q("#rbHarnessGenerations")) q("#rbHarnessGenerations").value = p.harness.generations || "";
        if(q("#rbHarnessSteps")) q("#rbHarnessSteps").value = p.harness.steps || "";
        if(q("#rbHarnessPop")) q("#rbHarnessPop").value = p.harness.pop || "";
        if(q("#rbHarnessGrid")) q("#rbHarnessGrid").value = p.harness.grid_size || "";
        if(q("#rbHarnessStride")) q("#rbHarnessStride").value = p.harness.stride || "";
        if(q("#rbHarnessCkptEvery")) q("#rbHarnessCkptEvery").value = p.harness.ckpt_every || "";
        if(q("#rbHarnessCkptKeep")) q("#rbHarnessCkptKeep").value = p.harness.ckpt_keep || "";
        if(q("#rbHarnessArtifacts")) q("#rbHarnessArtifacts").value = p.harness.expected_artifacts || "";
        if(q("#rbHarnessDefaultKnobs")) q("#rbHarnessDefaultKnobs").value = p.harness.default_knobs || "";

        // Axes replace
        const axesBox = q("#rbAxes");
        if(axesBox){ axesBox.innerHTML = ""; }
        const axes = Array.isArray(p.harness.axes) ? p.harness.axes : [];
        if(axes.length){
          for(const ax of axes){
            _builderAddAxisRow(ax?.name||"axis", Array.isArray(ax?.values)?ax.values:[]);
          }
        }else{
          _builderAddAxisRow("case", []);
        }
        Builder.axisFlags = (p.harness.axis_flags && typeof p.harness.axis_flags === "object") ? p.harness.axis_flags : {};
        _builderRefreshCaseDropdown();
        _builderRenderFlagsForSelectedCase();
      }
      // Manifest
      if(p.manifest){
        if(q("#rbManifestName")) q("#rbManifestName").value = p.manifest.name || "";
        if(q("#rbManifestOutRoot")) q("#rbManifestOutRoot").value = p.manifest.output_root || "";
        if(q("#rbManifestStrict")) q("#rbManifestStrict").checked = !!p.manifest.strict;
        if(q("#rbManifestGitRef")) q("#rbManifestGitRef").value = p.manifest.git_ref_override || "";
        if(q("#rbManifestManual")) q("#rbManifestManual").value = p.manifest.manual || "";
      }

      _builderClearHarnessBaseline();
      Builder.harnessLaunchReady = false;
      _builderSetRegistrySelection("");
      _builderRenderCliPreview();
      _builderRenderHarnessCmdPreview();
      _builderRefreshHarnessDirtyState();
      _builderRefreshRegistryActionState();
      _builderPersistNow();
      showToast("Template loaded", name);
    }catch(e){
      showToast("Load failed", e?.message||String(e));
    }
  });
}

function encRunKey(runKey){
  return encodeURIComponent(String(runKey || ""));
}

// -----------------------------
// Formatting
// -----------------------------

function detBadgeClass(det){
  const d = String(det || "").toUpperCase();
  if(d === "PASS") return "b-pass";
  if(d === "FAIL") return "b-fail";
  return "b-unknown";
}

// -----------------------------
// Global app state
// -----------------------------

const State = {
  runs: [],            // full runs index (include archived)
  changesets: [],
  lastRunsUtc: null,
  lastChangeSetsUtc: null,
  planChangeSets: new Set(),

  // Operational plane runtime
  systemRuns: null,
  lastSystemRunsUtc: null,
  runControl: {
    harnesses: [],
    lastHarnessesUtc: null,
    selectedHarnessModule: "",
    recentJobs: [],
    activeJobId: "",
    launchStatus: null,
    launchLogText: "",
    lastLaunchUtc: null,
    processes: [],
    lastProcessesUtc: null,
    includeAllProcesses: false,
    processBusyPid: 0,
    processBusyAction: "",
  },
  pressureSearch: {
    campaigns: [],
    activeCampaignId: "pressure_search_001",
    activeCampaign: null,
    lastLoadedUtc: null,
    busyAction: "",
    noteDraft: "",
    durationDays: 28,
    maxActiveChildren: 1,
    pollSeconds: 60,
    lastLaunchStatus: null,
  },

  // Patch backlog
  patchBacklog: null,
  lastPatchBacklogUtc: null,
  backlogDraft: {
    md: "",
    lastSavedUtc: null,
  },
  patchBacklogSig: null,
  patchBacklogRenderedSig: null,
  backlogUi: {
    expandedIds: new Set(),
    scrollTops: {},
  },
  // Backlog multi-select (desktop)
  backlogSelections: new Set(),
  // Legacy governance wrappers retained only for dormant compatibility helpers.
  backlogMode: "ledger", // "ledger" (default) | "legacy"
  ledger: {
    items: null,
    lastItemsUtc: null,
    itemsSig: null,
    itemsRenderedSig: null,
    activeUid: null,
    activeItem: null,       // full item payload from /api/ledger/item/<uid>
    activeLoadedUtc: null,
    editor: {
      pisDraft: "",
      metaDraft: null,
      dirtyPis: false,
      dirtyMeta: false,
    },
    ui: {
      search: "",
      status: "ALL",
      tag: "",
      showDeprecated: false,
      showReopened: true,
      listScrollTop: 0,
    }
  },

  // View and filters
  view: "overview",      // overview|matrix|changesets|builder|backlog|run
  activeRunKey: null,

  overview: {
    showArchived: false,
    search: "",
    onlyDetPass: false,
    filtersOpen: false,
    runControlOpen: true,
    launchLogOpen: false,
    processesOpen: false,
    systemRunsOpen: false,
    selectedRunOpen: true,
    lineageOpen: {},
    processTreeOpen: {},
    selectedRealRunKey: "",
    selectedRunStdoutKey: "",
    selectedRunStdoutText: "",
    selectedRunStdoutLoadedUtc: null,
    selectedRunStdoutLoading: false,
    selectedRunStdoutScrollTop: 0,
    selectedRunStdoutStickToBottom: true,
    launchLogScrollTop: 0,
  },
  matrix: {
    showArchived: false,
    search: "",
    status: "ALL",
    determinism: "ALL",
    strict: "ALL",
    tag: "",
    onlyDetPass: false,
    sortKey: "run",
    sortDir: 1,
  },
  changesetsView: {
    search: "",
  },
  runDetail: {
    range: 1000,
    overlayParent: true,
    overlayStatus: null,
    smoothEpi: true,
    showAdvanced: false,
    pendingUuids: new Set(),
    lastLoadedRunKey: null,
    // Incremental EPI/AV tail state (client-side). Server already tails safely.
    epiavLastGen: -1,
    epiavSource: null,
    parentEpiavLastGen: -1,
    parentEpiavSource: null,
    lastRange: 1000,
    lastParentRunKey: null,
    meta: null,
    epiav: null,
    parentEpiav: null,
    checkpoints: null,
    appliedCommands: null,
    appliedLimit: 200,
    appliedAfterGen: -1,
    appliedBeforeGen: null,
    appliedSearch: "",
    appliedStatus: "ALL",
    appliedWindow: null,
    inboxQueue: null,
    notes: null,
    // Autopilot (optional): autopilot/pressure_autopilot_decisions.jsonl
    autopilotActions: null,
    autopilotLimit: 200,
    autopilotAfterGen: -1,
    autopilotBeforeGen: null,
    autopilotSearch: "",
    autopilotWindow: null,
    // Metrics (optional): metrics.csv / viz/metrics_summary.csv
    metricsSchema: null,
    metricName: "alive",
    metricLastGen: -1,
    metricSeries: null,
    metricSource: null,
    lastMetricName: null,
    // Knobs (optional): run_config.csv
    knobs: null,
    knobsSource: null,
    // Quick action: PressureUpdateCommand helper UI state
    // NOTE: target gen default is set after meta loads (current_gen + 10).
    // Keep null here so we can distinguish "never touched" from a real user choice.
    pressureAutopilotEnable: true,
    pressureAutopilotTargetGen: null,
    pressureAutopilotTargetGenTouched: false,
    pressureAutopilotTickInterval: 200,
    pressureAutopilotProfile: "balanced",
    pressureAutopilotDeltaUp: 0.05,
    pressureAutopilotDeltaDown: 0.10,
    pressureAutopilotTargetSurvival: 0.55,
    pressureAutopilotSurvivalCeiling: 0.80,
    lastError: null,
    // Timeline (control + pressure) summary
    timelineBins: 200,
    timelineSummary: null,
    timelineLastSig: null,
    timelineSelection: null,
  },

  // Polling bookkeeping
  tick: 0,
  inFlight: false,
  refreshToken: 0,
};

// -----------------------------
// API wrappers
// -----------------------------

async function apiHealth(){
  return await ((__api.apiHealth || _missingApi("apiHealth"))());
}

async function apiRuns(){
  return await ((__api.apiRuns || _missingApi("apiRuns"))());
}

async function apiPressureSearchCampaigns(){
  return await ((__api.apiPressureSearchCampaigns || _missingApi("apiPressureSearchCampaigns"))());
}

async function apiPressureSearchCampaign(campaignId){
  return await ((__api.apiPressureSearchCampaign || _missingApi("apiPressureSearchCampaign"))(campaignId));
}

async function apiPressureSearchControl(payload){
  return await ((__api.apiPressureSearchControl || _missingApi("apiPressureSearchControl"))(payload));
}

async function apiPressureSearchStep(payload){
  return await ((__api.apiPressureSearchStep || _missingApi("apiPressureSearchStep"))(payload));
}

async function apiChangeSets(){
  return await ((__api.apiChangeSets || _missingApi("apiChangeSets"))());
}

async function apiSystemRuns(){
  return await ((__api.apiSystemRuns || _missingApi("apiSystemRuns"))());
}

async function apiPatchBacklog(){
  return await ((__api.apiPatchBacklog || _missingApi("apiPatchBacklog"))());
}

async function apiPatchBacklogSave(doc){
  return await ((__api.apiPatchBacklogSave || _missingApi("apiPatchBacklogSave"))(doc));
}


async function apiLedgerItems(limit){
  return await ((__api.apiLedgerItems || _missingApi("apiLedgerItems"))(limit));
}

async function apiLedgerItem(uid){
  return await ((__api.apiLedgerItem || _missingApi("apiLedgerItem"))(uid));
}

async function apiLedgerCreate(payload){
  return await ((__api.apiLedgerCreate || _missingApi("apiLedgerCreate"))(payload));
}

async function apiLedgerStatus(uid, status, note){
  return await ((__api.apiLedgerStatus || _missingApi("apiLedgerStatus"))(uid, status, note));
}

async function apiLedgerAdvance(uid, payload){
  return await ((__api.apiLedgerAdvance || _missingApi("apiLedgerAdvance"))(uid, payload));
}

async function apiLedgerUpdateMeta(uid, meta){
  return await ((__api.apiLedgerUpdateMeta || _missingApi("apiLedgerUpdateMeta"))(uid, meta));
}

async function apiLedgerUpdatePis(uid, pis_markdown){
  return await ((__api.apiLedgerUpdatePis || _missingApi("apiLedgerUpdatePis"))(uid, pis_markdown));
}

async function apiLedgerCreateRevision(uid, payload){
  return await ((__api.apiLedgerCreateRevision || _missingApi("apiLedgerCreateRevision"))(uid, payload));
}

async function apiLedgerCommits(uid, limit){
  return await ((__api.apiLedgerCommits || _missingApi("apiLedgerCommits"))(uid, limit));
}

async function apiLedgerActionRun(payload){
  return await ((__api.apiLedgerActionRun || _missingApi("apiLedgerActionRun"))(payload));
}

async function apiLedgerActionStatus(uid, jobId, limit){
  return await ((__api.apiLedgerActionStatus || _missingApi("apiLedgerActionStatus"))(uid, jobId, limit));
}

async function apiLedgerActionLog(uid, jobId, lines){
  return await ((__api.apiLedgerActionLog || _missingApi("apiLedgerActionLog"))(uid, jobId, lines));
}

async function apiLedgerActionCancel(uid, jobId){
  return await ((__api.apiLedgerActionCancel || _missingApi("apiLedgerActionCancel"))(uid, jobId));
}

async function apiLedgerActionForceCancel(uid, jobId){
  return await ((__api.apiLedgerActionForceCancel || _missingApi("apiLedgerActionForceCancel"))(uid, jobId));
}

async function apiLedgerMarkVerified(uid){
  return await ((__api.apiLedgerMarkVerified || _missingApi("apiLedgerMarkVerified"))(uid));
}

async function apiAgentsSendTo(payload){
  return await ((__api.apiAgentsSendTo || _missingApi("apiAgentsSendTo"))(payload));
}

async function apiAgentsHandoff(payload){
  return await ((__api.apiAgentsHandoff || _missingApi("apiAgentsHandoff"))(payload));
}

async function apiAgentsSession(role, binding){
  return await ((__api.apiAgentsSession || _missingApi("apiAgentsSession"))(role, binding));
}

async function apiGitStatus(){
  return await ((__api.apiGitStatus || _missingApi("apiGitStatus"))());
}

async function apiBacklogGptSubmit(payload){
  return await ((__api.apiBacklogGptSubmit || _missingApi("apiBacklogGptSubmit"))(payload));
}

async function apiRunMeta(runKey){
  return await ((__api.apiRunMeta || _missingApi("apiRunMeta"))(runKey));
}

async function apiRunEpiAv(runKey, afterGen, limit){
  return await ((__api.apiRunEpiAv || _missingApi("apiRunEpiAv"))(runKey, afterGen, limit));
}

async function apiRunCheckpoints(runKey){
  return await ((__api.apiRunCheckpoints || _missingApi("apiRunCheckpoints"))(runKey));
}

async function apiRunAppliedCommands(runKey, opts){
  return await ((__api.apiRunAppliedCommands || _missingApi("apiRunAppliedCommands"))(runKey, opts));
}



async function apiRunAutopilotActions(runKey, opts){
  return await ((__api.apiRunAutopilotActions || _missingApi("apiRunAutopilotActions"))(runKey, opts));
}


async function apiRunTimelineSummary(runKey, bins){
  return await ((__api.apiRunTimelineSummary || _missingApi("apiRunTimelineSummary"))(runKey, bins));
}

async function apiRunTimelineRange(runKey, genMin, genMax, kind){
  return await ((__api.apiRunTimelineRange || _missingApi("apiRunTimelineRange"))(runKey, genMin, genMax, kind));
}


async function apiRunInboxQueue(runKey, limit){
  return await ((__api.apiRunInboxQueue || _missingApi("apiRunInboxQueue"))(runKey, limit));
}



async function apiRunMetricsSchema(runKey){
  return await ((__api.apiRunMetricsSchema || _missingApi("apiRunMetricsSchema"))(runKey));
}

async function apiRunMetricSeries(runKey, metric, afterGen, limit){
  return await ((__api.apiRunMetricSeries || _missingApi("apiRunMetricSeries"))(runKey, metric, afterGen, limit));
}

async function apiRunTraits(runKey){
  return await ((__api.apiRunTraits || _missingApi("apiRunTraits"))(runKey));
}

async function apiRunKnobs(runKey){
  return await ((__api.apiRunKnobs || _missingApi("apiRunKnobs"))(runKey));
}
async function apiRunNotes(runKey){
  return await ((__api.apiRunNotes || _missingApi("apiRunNotes"))(runKey));
}

async function apiWriteNotes(runKey, text){
  return await ((__api.apiWriteNotes || _missingApi("apiWriteNotes"))(runKey, text));
}

async function apiArchive(runKey, archived){
  return await ((__api.apiArchive || _missingApi("apiArchive"))(runKey, archived));
}

// Full run archive zip (stored under runtime/artifacts/archives).
async function apiArchiveRun(runKey, opts){
  return await ((__api.apiArchiveRun || _missingApi("apiArchiveRun"))(runKey, opts));
}
async function apiUnarchiveRun(runKey){
  return await ((__api.apiUnarchiveRun || _missingApi("apiUnarchiveRun"))(runKey));
}



async function apiWriteUiMeta(runKey, patch){
  return await ((__api.apiWriteUiMeta || _missingApi("apiWriteUiMeta"))(runKey, patch));
}

async function apiExport(runKey, includeCheckpoints){
  return await ((__api.apiExport || _missingApi("apiExport"))(runKey, includeCheckpoints));
}

// Run analysis snapshot export (UI-owned export directory).
// Supports optional name + include_checkpoints flag.
async function apiExportSnapshot(runKey, name, includeCheckpoints){
  return await ((__api.apiExportSnapshot || _missingApi("apiExportSnapshot"))(runKey, name, includeCheckpoints));
}

async function apiSubmitCommand(runKey, payload){
  return await ((__api.apiSubmitCommand || _missingApi("apiSubmitCommand"))(runKey, payload));
}

async function apiPlanBranch(payload){
  return await ((__api.apiPlanBranch || _missingApi("apiPlanBranch"))(payload));
}
