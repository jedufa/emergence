"use strict";

const $ = (selector) => document.querySelector(selector);
let selectedKey = null;
let latestSlots = new Map();
const ACTIVE_SLOT_STATUSES = new Set(["starting", "running", "finishing", "stale", "conflict", "untracked"]);
const TERMINAL_SLOT_STATUSES = new Set(["complete", "failed", "interrupted", "cancelled"]);
const REFRESH_INTERVAL_MS = 2000;

function text(value, fallback = "—") {
  return value === null || value === undefined || value === "" ? fallback : String(value);
}

function html(value, fallback = "—") {
  return text(value, fallback).replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
  })[character]);
}

function duration(seconds) {
  if (!Number.isFinite(Number(seconds))) return "Learning…";
  const s = Math.max(0, Math.round(Number(seconds)));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60), rem = s % 60;
  if (m < 60) return `${m}m ${rem}s`;
  const h = Math.floor(m / 60), min = m % 60;
  return `${h}h ${min}m`;
}

function speed(rate) {
  const n = Number(rate);
  if (!Number.isFinite(n) || n <= 0) return "Learning…";
  return n >= 0.1 ? `${n.toFixed(2)} gen/s` : `${(n * 60).toFixed(2)} gen/min`;
}

function completionTime(eta) {
  const n = Number(eta);
  if (!Number.isFinite(n)) return "Learning…";
  return new Date(Date.now() + n * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
}

function slotKey(slot) { return `${slot.device}:${slot.position}`; }

function canopyAvailable(slot) {
  return Boolean(slot && slot.observatory && slot.observatory.canopy_available && slot.observatory.canopy_url);
}

function openCanopy(slot) {
  if (canopyAvailable(slot)) window.open(slot.observatory.canopy_url, "_blank", "noopener");
}

function orderedSlots(slots) {
  const rank = slot => {
    if (ACTIVE_SLOT_STATUSES.has(slot.status)) return 0;
    if (TERMINAL_SLOT_STATUSES.has(slot.status)) return 1;
    return 2;
  };
  return [...slots].sort((left, right) => rank(left) - rank(right) || left.position - right.position);
}

function slotMarkup(slot) {
  const fraction = Math.max(0, Math.min(1, Number(slot.progress_fraction) || 0));
  const generation = slot.current_generation === null ? "—" : `${slot.current_generation}/${slot.target_generation}`;
  const idle = slot.status === "idle";
  const canopy = canopyAvailable(slot);
  return `<button class="slot ${html(slot.status)}" data-key="${html(slotKey(slot))}" type="button">
    <div class="slot-top"><span class="slot-name">${html(slot.label)}</span><span class="status">${html(slot.status)}</span></div>
    <div class="cell">${idle ? "Available" : html(slot.cell_id)}</div>
    <div class="bar"><i style="width:${(fraction * 100).toFixed(2)}%"></i></div>
    <div class="slot-stats"><span>Gen ${generation}</span><span>${idle ? "" : duration(slot.eta_seconds)}</span><span>${idle ? "" : speed(slot.rate_generations_per_second)}</span><span>${idle ? "" : `ETA ${completionTime(slot.eta_seconds)}`}</span></div>
    ${canopy ? `<span class="canopy-chip" aria-hidden="true">Canopy available</span>` : ""}
  </button>`;
}

function renderGrid(selector, slots) {
  const root = $(selector);
  root.innerHTML = orderedSlots(slots).map(slotMarkup).join("");
  root.querySelectorAll(".slot").forEach(button => button.addEventListener("click", () => showDetails(button.dataset.key)));
}

function showDetails(key, shouldScroll = true) {
  selectedKey = key;
  const slot = latestSlots.get(key);
  if (!slot) return;
  $("#details").classList.remove("hidden");
  $("#detailTitle").textContent = `${slot.label} · ${slot.status}`;
  const rows = [
    ["Cell", slot.cell_id], ["Run", slot.run_id], ["Campaign", slot.campaign_id], ["Batch", slot.batch_id],
    ["Stage", slot.stage], ["Progress", slot.current_generation === null ? null : `${slot.current_generation} / ${slot.target_generation}`],
    ["Speed", speed(slot.rate_generations_per_second)], ["ETA", `${duration(slot.eta_seconds)} · ${completionTime(slot.eta_seconds)}`],
    ["Elapsed", duration(slot.elapsed_seconds)], ["PID", slot.pid], ["Updated", slot.updated_at], ["Run root", slot.run_root],
    ["Canopy", canopyAvailable(slot) ? `Available · ${slot.observatory.semantics || "playback"}` : "Unavailable"],
  ];
  if (slot.error) rows.push(["Error", slot.error]);
  if (slot.conflicts && slot.conflicts.length) rows.push(["Conflicts", JSON.stringify(slot.conflicts)]);
  const action = canopyAvailable(slot)
    ? `<div class="detail detail-action"><button id="openCanopy" type="button">Open Canopy</button><span>Read-only lineage playback for this slot.</span></div>`
    : "";
  $("#detailBody").innerHTML = `${action}${rows.map(([label, value]) => `<div class="detail"><label>${html(label)}</label><div>${html(value)}</div></div>`).join("")}`;
  const canopyButton = $("#openCanopy");
  if (canopyButton) canopyButton.addEventListener("click", () => openCanopy(slot));
  if (shouldScroll) $("#details").scrollIntoView({behavior: "smooth", block: "nearest"});
}

function setSummary(data) {
  $("#gpuCount").textContent = `${data.gpu_active} / ${data.gpu_capacity}`;
  $("#cpuCount").textContent = `${data.cpu_active} / ${data.cpu_capacity}`;
  $("#gpuMeter").style.width = `${100 * data.gpu_active / data.gpu_capacity}%`;
  $("#cpuMeter").style.width = `${100 * data.cpu_active / data.cpu_capacity}%`;
  const gpu = data.nvidia || {};
  $("#gpuName").textContent = text(gpu.name, "NVIDIA unavailable");
  $("#gpuUtil").textContent = Number.isFinite(gpu.utilization_percent) ? `${gpu.utilization_percent}% load` : "—% load";
  $("#gpuMemory").textContent = Number.isFinite(gpu.memory_used_mib) ? `${Math.round(gpu.memory_used_mib)} / ${Math.round(gpu.memory_total_mib)} MiB` : "— VRAM";
  $("#gpuTemp").textContent = Number.isFinite(gpu.temperature_c) ? `${gpu.temperature_c}°C` : "—°C";
  $("#lastUpdate").textContent = `Updated ${new Date().toLocaleTimeString()}`;
  const issues = [];
  const missingGpuTelemetry = (data.untracked_gpu_processes || []).filter(row => !row.generation_progress_available);
  if (missingGpuTelemetry.length) issues.push(`Untracked GPU activity: ${missingGpuTelemetry.map(row => `${row.process} (PID ${row.pid})`).join(", ")}`);
  const untrackedEmergence = (data.untracked_emergence_processes || []).filter(row => !(data.untracked_gpu_processes || []).some(gpu => gpu.pid === row.pid));
  if (untrackedEmergence.length) issues.push(`Untracked Emergence process: ${untrackedEmergence.map(row => `${row.process} (PID ${row.pid})`).join(", ")}`);
  if ((data.invalid_heartbeats || []).length) issues.push(`${data.invalid_heartbeats.length} invalid telemetry file(s)`);
  const conflicts = [...data.gpu_slots, ...data.cpu_slots].filter(row => row.status === "conflict");
  if (conflicts.length) issues.push(`Allocation conflict on ${conflicts.map(row => row.label).join(", ")}`);
  const warning = $("#warning");
  warning.textContent = issues.join(" · ");
  warning.classList.toggle("hidden", issues.length === 0);
}

async function refresh() {
  try {
    const response = await fetch("/api/v1/snapshot", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    latestSlots = new Map([...data.gpu_slots, ...data.cpu_slots].map(slot => [slotKey(slot), slot]));
    renderGrid("#gpuGrid", data.gpu_slots);
    renderGrid("#cpuGrid", data.cpu_slots);
    setSummary(data);
    const connection = $("#connection");
    connection.className = "connection online";
    connection.innerHTML = "<span></span> Live · 2s";
    if (selectedKey) showDetails(selectedKey, false);
  } catch (error) {
    const connection = $("#connection");
    connection.className = "connection offline";
    connection.innerHTML = "<span></span> Offline";
    $("#warning").textContent = `Monitor unavailable: ${error.message}`;
    $("#warning").classList.remove("hidden");
  }
}

$("#detailClose").addEventListener("click", () => { selectedKey = null; $("#details").classList.add("hidden"); });
refresh();
setInterval(refresh, REFRESH_INTERVAL_MS);
