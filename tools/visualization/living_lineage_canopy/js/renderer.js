import { hsla } from "./colors.js";

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

export class Renderer {
  constructor(canvas, system) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: true });
    this.system = system;
    this.dpr = 1;
    this.lastHitParticles = [];
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    this.canvas.width = Math.floor(rect.width * this.dpr);
    this.canvas.height = Math.floor(rect.height * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.system.resize(rect.width, rect.height);
  }

  render({ selectedLineageId = null, selectedAgentId = null, isolated = false, highDensity = false } = {}) {
    const start = performance.now();
    const ctx = this.ctx;
    const w = this.system.width;
    const h = this.system.height;
    ctx.clearRect(0, 0, w, h);
    this.drawLiquidField(ctx, w, h);
    this.drawSuspendedMotes(ctx, w, h);
    if (selectedLineageId) this.drawLineageFocus(ctx, selectedLineageId, isolated);
    this.drawDensity(ctx, selectedLineageId, isolated);

    const drawList = [
      ...this.system.deadParticles.map((p) => ({ particle: p, dying: true })),
      ...this.system.particles.values()].map((p) => p.particle ? p : ({ particle: p, dying: false }));
    drawList.sort((a, b) => (a.particle.lineageId > b.particle.lineageId ? 1 : -1));
    this.lastHitParticles = [];
    for (const item of drawList) {
      this.drawParticle(ctx, item.particle, {
        dying: item.dying,
        selected: item.particle.id === selectedAgentId,
        related: selectedLineageId && item.particle.lineageId === selectedLineageId,
        muted: selectedLineageId && item.particle.lineageId !== selectedLineageId,
        dimmed: isolated && selectedLineageId && item.particle.lineageId !== selectedLineageId,
        highDensity,
      });
      if (!item.dying) this.lastHitParticles.push(item.particle);
    }
    return performance.now() - start;
  }

  drawLiquidField(ctx, w, h) {
    const gradient = ctx.createRadialGradient(w * 0.48, h * 0.48, 40, w * 0.48, h * 0.52, Math.max(w, h) * 0.7);
    gradient.addColorStop(0, "rgba(22, 39, 40, 0.42)");
    gradient.addColorStop(0.48, "rgba(8, 15, 17, 0.16)");
    gradient.addColorStop(1, "rgba(0, 0, 0, 0.42)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, w, h);

    const lateral = ctx.createLinearGradient(0, h * 0.18, w, h * 0.92);
    lateral.addColorStop(0, "rgba(98, 145, 135, 0.035)");
    lateral.addColorStop(0.5, "rgba(242, 236, 214, 0.018)");
    lateral.addColorStop(1, "rgba(132, 94, 122, 0.026)");
    ctx.fillStyle = lateral;
    ctx.fillRect(0, 0, w, h);
  }

  drawSuspendedMotes(ctx, w, h) {
    ctx.save();
    ctx.globalCompositeOperation = "screen";
    const count = clamp(Math.floor((w * h) / 30000), 22, 76);
    for (let i = 0; i < count; i += 1) {
      const x = ((Math.sin(i * 83.17) * 0.5 + 0.5) * 0.92 + 0.04) * w;
      const y = ((Math.sin(i * 47.39 + 1.7) * 0.5 + 0.5) * 0.86 + 0.07) * h;
      const radius = 0.34 + ((i * 17) % 13) * 0.075;
      ctx.fillStyle = `rgba(208, 229, 223, ${0.016 + (i % 5) * 0.004})`;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  drawLineageFocus(ctx, selectedLineageId, isolated) {
    const branches = [];
    for (const [branchId, center] of this.system.branchCenters) {
      if (this.system.branchLineages.get(branchId) === selectedLineageId) {
        branches.push({ branchId, center, color: this.system.branchColors.get(branchId) });
      }
    }
    if (!branches.length) return;

    ctx.save();
    ctx.globalCompositeOperation = "screen";
    for (const branch of branches) {
      const radius = clamp(Math.sqrt(branch.center.n) * 8.3, 38, 168);
      const gradient = ctx.createRadialGradient(branch.center.x, branch.center.y, 2, branch.center.x, branch.center.y, radius);
      gradient.addColorStop(0, hsla(branch.color, isolated ? 0.24 : 0.2, 14));
      gradient.addColorStop(0.5, hsla(branch.color, isolated ? 0.11 : 0.09, 2));
      gradient.addColorStop(1, hsla(branch.color, 0, -10));
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(branch.center.x, branch.center.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  drawDensity(ctx, selectedLineageId, isolated) {
    ctx.save();
    ctx.globalCompositeOperation = "screen";
    for (const [branchId, center] of this.system.branchCenters) {
      if (center.n < 8) continue;
      const color = this.system.branchColors.get(branchId);
      const lineageId = this.system.branchLineages.get(branchId);
      if (!lineageId) continue;
      const dim = selectedLineageId && lineageId !== selectedLineageId;
      const radius = clamp(Math.sqrt(center.n) * 6.6, 26, 134);
      const gradient = ctx.createRadialGradient(center.x, center.y, 3, center.x, center.y, radius);
      gradient.addColorStop(0, hsla(color, dim ? 0.018 : 0.105, 6));
      gradient.addColorStop(0.58, hsla(color, dim ? 0.004 : 0.03, -2));
      gradient.addColorStop(1, hsla(color, 0, 0));
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  drawParticle(ctx, particle, flags) {
    const deathAlpha = flags.dying ? clamp(1 - particle.deathAge / 0.85, 0, 1) : 1;
    const dimAlpha = flags.dimmed ? 0.052 : flags.muted ? 0.18 : 1;
    const relatedLift = flags.related ? 1.3 : 1;
    const selectedLift = flags.selected ? 1.7 : 1;
    const radius = this.system.visualRadius(particle) * relatedLift * selectedLift * (flags.dying ? deathAlpha : 1);
    const alpha = deathAlpha * dimAlpha * (flags.related ? 1 : 0.76);
    if (radius <= 0.1 || alpha <= 0.01) return;

    ctx.save();
    ctx.globalCompositeOperation = "screen";
    if (flags.selected) {
      const halo = ctx.createRadialGradient(particle.x, particle.y, radius, particle.x, particle.y, radius * 5.6);
      halo.addColorStop(0, hsla(particle.color, 0.28, 10));
      halo.addColorStop(1, hsla(particle.color, 0, 0));
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(particle.x, particle.y, radius * 5.6, 0, Math.PI * 2);
      ctx.fill();
    }
    if (flags.highDensity && !flags.selected) {
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = hsla(particle.color, alpha * 0.72, 1);
      ctx.beginPath();
      ctx.arc(particle.x, particle.y, radius * 0.9, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      return;
    }

    const gradient = ctx.createRadialGradient(
      particle.x - radius * 0.25,
      particle.y - radius * 0.3,
      radius * 0.15,
      particle.x,
      particle.y,
      radius * 1.8,
    );
    gradient.addColorStop(0, hsla(particle.color, alpha, 15));
    gradient.addColorStop(0.45, hsla(particle.color, alpha * 0.78, 0));
    gradient.addColorStop(1, hsla(particle.color, 0, -12));
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(particle.x, particle.y, radius * 1.85, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = hsla(particle.color, alpha * 0.92, 2);
    ctx.beginPath();
    ctx.arc(particle.x, particle.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  hitTest(x, y) {
    let best = null;
    let bestDistance = Infinity;
    for (let i = this.lastHitParticles.length - 1; i >= 0; i -= 1) {
      const particle = this.lastHitParticles[i];
      const r = Math.max(7, this.system.visualRadius(particle) * 2.2);
      const dx = x - particle.x;
      const dy = y - particle.y;
      const d2 = dx * dx + dy * dy;
      if (d2 < r * r && d2 < bestDistance) {
        best = particle;
        bestDistance = d2;
      }
    }
    return best;
  }
}
