"use strict";

(function(){
  const REFRESH_MS = 2000;
  const ACTIVE = new Set(["starting", "running", "finishing", "stale", "conflict", "untracked"]);
  const TERMINAL = new Set(["complete", "failed", "interrupted", "cancelled"]);
  let timer = null;
  let inFlight = false;
  let selectedKey = null;
  let slotsByKey = new Map();
  let chronicleRequest = 0;

  const root = () => document.querySelector("#view-live-simulations");
  const byId = id => document.getElementById(id);
  const esc = value => String(value == null || value === "" ? "—" : value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[c]);
  const key = slot => `${slot.device}:${slot.position}`;

  function duration(seconds){
    if(!Number.isFinite(Number(seconds))) return "Learning…";
    const total = Math.max(0, Math.round(Number(seconds)));
    if(total < 60) return `${total}s`;
    const minutes = Math.floor(total / 60), remaining = total % 60;
    if(minutes < 60) return `${minutes}m ${remaining}s`;
    return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
  }

  function speed(rate){
    const value = Number(rate);
    if(!Number.isFinite(value) || value <= 0) return "Learning…";
    if(value >= 100) return `${Math.round(value)} tick/s`;
    return `${value.toFixed(2)} tick/s`;
  }

  function tickProgress(slot){
    const next = Number(slot.next_absolute_tick);
    const target = Number(slot.target_absolute_tick);
    return Number.isFinite(next) && Number.isFinite(target)
      ? `${next}/${target}${slot.tick_precision === "boundary-only" ? " (boundary)" : ""}`
      : "telemetry unavailable";
  }

  function completionTime(eta){
    const value = Number(eta);
    if(!Number.isFinite(value)) return "Learning…";
    return new Date(Date.now() + value * 1000).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
  }

  function percent(value){
    const number = Number(value);
    return Number.isFinite(number) ? `${number.toFixed(0)}%` : "Learning…";
  }

  function canopyAvailable(slot){
    return Boolean(slot && slot.observatory && slot.observatory.canopy_available && slot.observatory.canopy_url);
  }

  function canopyStatus(slot){
    if(canopyAvailable(slot)) return `Available · ${slot.observatory.semantics || "playback"}`;
    const observatory = slot && slot.observatory;
    if(!observatory || !observatory.canopy_reason) return "Unavailable";
    const cadence = Number(observatory.canopy_checkpoint_every_ticks);
    const suffix = Number.isFinite(cadence) && cadence > 0 ? ` · checkpoint every ${cadence} ticks` : "";
    return `Unavailable · ${observatory.canopy_reason}${suffix}`;
  }

  function openCanopy(slot){
    if(!canopyAvailable(slot)) return;
    window.open(slot.observatory.canopy_url, "_blank", "noopener");
  }

  function memory(value){
    const number = Number(value);
    return Number.isFinite(number) ? `${Math.round(number)} MiB` : "—";
  }

  function ordered(slots){
    const rank = slot => ACTIVE.has(slot.status) ? 0 : TERMINAL.has(slot.status) ? 1 : 2;
    const deviceRank = slot => slot.device === "cpu" ? 0 : 1;
    return [...slots].sort((left, right) => rank(left) - rank(right) || deviceRank(left) - deviceRank(right) || left.position - right.position);
  }

  function shell(){
    const target = root();
    if(!target || target.dataset.ready === "1") return;
    target.dataset.ready = "1";
    target.innerHTML = `
      <div class="ls-page">
        <section class="ls-section"><div class="ls-section-head"><div><div class="ls-kicker">CPU + PHYSICAL RESIDENT GPU BATCH</div><h2>Simulation slots</h2></div><div class="ls-section-meta"><div class="ls-legend"><span>Active first</span><span>CPU blue</span><span>Idle last</span></div><div id="lsConnection" class="ls-connection" aria-live="polite">Connecting</div></div></div><div id="lsHardware" class="ls-hardware" aria-live="polite"></div><div id="lsSlotGrid" class="ls-grid"></div></section>
        <section id="lsDetails" class="ls-details hidden"><div class="ls-detail-head"><div><div class="ls-kicker">SELECTED WORLD</div><h2 id="lsDetailTitle">World details</h2></div><button id="lsDetailClose" class="btn btn-ghost" type="button">Close</button></div><div id="lsDetailBody" class="ls-detail-grid"></div><section id="lsChronicle" class="ls-chronicle" aria-live="polite"></section></section>
      </div>`;
    byId("lsDetailClose").addEventListener("click", () => { selectedKey = null; chronicleRequest += 1; byId("lsDetails").classList.add("hidden"); });
  }

  function slotMarkup(slot){
    const fraction = Math.max(0, Math.min(1, Number(slot.progress_fraction) || 0));
    const idle = slot.status === "idle";
    const ticks = tickProgress(slot);
    const canopy = canopyAvailable(slot);
    return `<button class="ls-slot ls-device-${esc(slot.device)} ls-${esc(slot.status)}" data-slot="${esc(key(slot))}" type="button">
      <div class="ls-slot-top"><strong>${esc(slot.label)}</strong><span>${esc(slot.status)}</span></div>
      <div class="ls-cell">${idle ? "Available" : esc(slot.cell_id)}</div>
      <div class="ls-bar"><i style="width:${(fraction * 100).toFixed(2)}%"></i></div>
      <div class="ls-stats"><span>Tick ${ticks}</span><span>${idle ? "" : duration(slot.eta_seconds)}</span><span>${idle ? "" : speed(slot.rate_ticks_per_second)}</span><span>${idle ? "" : `ETA ${completionTime(slot.eta_seconds)}`}</span></div>
      ${canopy ? `<span class="ls-canopy-chip" aria-hidden="true">Canopy available</span>` : ""}
    </button>`;
  }

  function showDetails(slotKey){
    selectedKey = slotKey;
    const slot = slotsByKey.get(slotKey);
    if(!slot) return;
    byId("lsDetails").classList.remove("hidden");
    byId("lsDetailTitle").textContent = `${slot.label} · ${slot.status}`;
    const rows = [
      ["Cell", slot.cell_id], ["Run", slot.run_id], ["Campaign", slot.campaign_id], ["Batch", slot.batch_id],
      ["Stage", slot.stage], ["Progress", `Tick ${tickProgress(slot)}`],
      ["Speed", speed(slot.rate_ticks_per_second)], ["ETA", `${duration(slot.eta_seconds)} · ${completionTime(slot.eta_seconds)}`],
      ["PID", slot.pid], ["Updated", slot.updated_at], ["Run root", slot.run_root], ["Source", slot.telemetry_source],
      ["Canopy", canopyStatus(slot)],
    ];
    if(slot.error) rows.push(["Error", slot.error]);
    const action = canopyAvailable(slot)
      ? `<div class="ls-detail-actions"><button id="lsOpenCanopy" class="btn" type="button">Open Canopy</button><p>Read-only lineage playback for this simulation slot.</p></div>`
      : "";
    byId("lsDetailBody").innerHTML = `${action}${rows.map(([label, value]) => `<div><label>${esc(label)}</label><p>${esc(value)}</p></div>`).join("")}`;
    const canopyButton = byId("lsOpenCanopy");
    if(canopyButton) canopyButton.addEventListener("click", () => openCanopy(slot));
    loadChronicle(slotKey, slot);
  }

  function chronicleItem(item){
    if(item.kind === "episode"){
      const episode = item.episode || {};
      return `<li><strong>Tick telemetry unavailable</strong><span>${esc(episode.title || episode.episode_type || "Episode")}</span></li>`;
    }
    const event = item.formal_event || {};
    const entry = item.entry || {};
    const tick = Number(event.next_absolute_tick);
    const coordinate = Number.isFinite(tick) ? `Tick ${tick}` : "Tick telemetry unavailable";
    return `<li><strong>${esc(coordinate)}</strong><span>${esc(entry.headline || event.event_type || "Event")}</span></li>`;
  }

  async function loadChronicle(slotKey, slot){
    const target = byId("lsChronicle");
    const request = ++chronicleRequest;
    if(slot.status === "idle"){
      target.innerHTML = `<div class="ls-chronicle-head"><div><div class="ls-kicker">WORLD CHRONICLE</div><h3>No world selected</h3></div></div><p class="ls-chronicle-note">This GPU position is available.</p>`;
      return;
    }
    target.innerHTML = `<div class="ls-chronicle-head"><div><div class="ls-kicker">WORLD CHRONICLE</div><h3>Loading Chronicle…</h3></div></div>`;
    try{
      const params = new URLSearchParams({device:String(slot.device), position:String(slot.position)});
      const response = await fetch(`/api/emergence/live_simulations/chronicle?${params}`, {cache:"no-store"});
      const data = await response.json();
      if(!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      if(request !== chronicleRequest || selectedKey !== slotKey) return;
      if(!["ready", "live", "archived"].includes(data.chronicle_status)){
        target.innerHTML = `<div class="ls-chronicle-head"><div><div class="ls-kicker">WORLD CHRONICLE</div><h3>${esc(data.chronicle_status)}</h3></div></div><p class="ls-chronicle-note">${esc(data.message)}</p>`;
        return;
      }
      const overview = data.overview || {};
      const items = (data.timeline && data.timeline.items) || [];
      const isLive = data.chronicle_status === "live";
      const isArchived = data.chronicle_status === "archived";
      const href = `/chronicle?run_key=${encodeURIComponent(data.run_key || "")}`;
      const action = isLive
        ? `<span class="ls-live-badge">LIVE / PROVISIONAL</span>`
        : `<div class="ls-chronicle-actions"><span class="ls-archive-badge">${isArchived ? "ARCHIVED" : "FINALIZED"}</span><a class="btn" href="${href}">Open full Chronicle</a></div>`;
      target.innerHTML = `<div class="ls-chronicle-head"><div><div class="ls-kicker">WORLD CHRONICLE</div><h3>${esc(overview.run_id || slot.cell_id)}</h3></div>${action}</div>
        <div class="ls-chronicle-stats"><span>${esc(overview.event_count)} events</span><span>${esc(overview.episode_count)} episodes</span><span>${esc(overview.source_completeness)}</span><span>${overview.finalized ? "Finalized" : "Provisional"}</span></div>
        ${isLive ? `<p class="ls-chronicle-note">${esc(data.message)}</p>` : ""}<ol class="ls-chronicle-list">${items.map(chronicleItem).join("") || "<li><span>No Chronicle events recorded.</span></li>"}</ol>`;
    }catch(error){
      if(request !== chronicleRequest || selectedKey !== slotKey) return;
      target.innerHTML = `<div class="ls-chronicle-head"><div><div class="ls-kicker">WORLD CHRONICLE</div><h3>Unavailable</h3></div></div><p class="ls-chronicle-note">${esc(error.message)}</p>`;
    }
  }

  function renderGrid(id, slots){
    const target = byId(id);
    target.innerHTML = ordered(slots).map(slotMarkup).join("");
    target.querySelectorAll("[data-slot]").forEach(button => button.addEventListener("click", () => showDetails(button.dataset.slot)));
  }

  function renderHardware(data){
    const cpu = data.cpu_system || {};
    const gpu = data.nvidia || {};
    const untracked = data.untracked_emergence_processes || [];
    const processText = untracked.slice(0, 2).map(row => `${row.process} · PID ${row.pid}`).join(", ");
    const extra = untracked.length > 2 ? ` +${untracked.length - 2}` : "";
    const target = byId("lsHardware");
    target.innerHTML = `
      <span><strong>CPU hardware</strong>${percent(cpu.utilization_percent)} · load ${Number(cpu.load_average_1m || 0).toFixed(2)} · ${esc(cpu.logical_cpu_count || "—")} threads</span>
      <span><strong>GPU hardware</strong>${percent(gpu.utilization_percent)} · ${memory(gpu.memory_used_mib)} / ${memory(gpu.memory_total_mib)}</span>
      <span><strong>Simulation cells</strong>${esc(data.cpu_active)}/${esc(data.cpu_capacity)} CPU · ${esc(data.gpu_active)}/${esc(data.gpu_capacity)} GPU</span>
      <span class="${untracked.length ? "ls-untracked-processes" : ""}"><strong>Untracked EIEL</strong>${untracked.length ? `${esc(untracked.length)} · ${esc(processText)}${esc(extra)}` : "None"}</span>`;
    target.title = untracked.length
      ? "These EIEL processes are running without simulation-slot telemetry. Hardware load and card occupancy can differ."
      : "Hardware load includes desktop and other non-simulation activity.";
  }

  function render(data){
    const slots = [...data.cpu_slots, ...data.gpu_slots];
    slotsByKey = new Map(slots.map(slot => [key(slot), slot]));
    renderHardware(data);
    renderGrid("lsSlotGrid", slots);
    byId("lsConnection").textContent = "Live · 2s";
    byId("lsConnection").className = "ls-connection online";
    byId("lsConnection").title = "Updated " + new Date().toLocaleTimeString();
    if(selectedKey) showDetails(selectedKey);
  }

  async function refresh(){
    if(inFlight || !root() || root().classList.contains("hidden")) return;
    inFlight = true;
    try{
      const response = await fetch("/api/emergence/live_simulations", {cache:"no-store"});
      if(!response.ok) throw new Error(`HTTP ${response.status}`);
      render(await response.json());
    }catch(error){
      const connection = byId("lsConnection");
      if(connection){ connection.textContent = "Offline"; connection.className = "ls-connection offline"; connection.title = error.message; }
    }finally{
      inFlight = false;
    }
  }

  function start(){
    shell();
    refresh();
    if(!timer) timer = setInterval(refresh, REFRESH_MS);
  }

  function stop(){
    if(timer) clearInterval(timer);
    timer = null;
  }

  window.LiveSimulationsView = {start, stop, refresh};
})();
