import { branchColor, hashString, lineageColor, mulberry32 } from "./colors.js";

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

function easeOutCubic(t) {
  return 1 - Math.pow(1 - clamp(t, 0, 1), 3);
}

function norm(dx, dy) {
  const d = Math.sqrt(dx * dx + dy * dy) || 0.0001;
  return { d, nx: dx / d, ny: dy / d };
}

export class ParticleSystem {
  constructor() {
    this.width = 1;
    this.height = 1;
    this.seed = 1;
    this.particles = new Map();
    this.deadParticles = [];
    this.lineageIndex = new Map();
    this.lineageColors = new Map();
    this.branchColors = new Map();
    this.branchLineages = new Map();
    this.branchParents = new Map();
    this.lineageBodies = new Map();
    this.branchBodies = new Map();
    this.branchCenters = new Map();
    this.lastTick = 0;
    this.stepIndex = 0;
    this.options = {
      particleScale: 1,
      motion: 0.72,
      branchPull: 1,
      lineagePull: 0.82,
      showBirthPulse: true,
      showDying: true,
    };
  }

  configure(playback) {
    this.seed = playback.metadata.visualization_seed || 1;
    this.lineageIndex.clear();
    this.lineageColors.clear();
    this.branchColors.clear();
    this.branchLineages.clear();
    this.branchParents.clear();
    playback.lineages.forEach((lineage, index) => {
      this.lineageIndex.set(lineage.lineage_id, index);
      this.lineageColors.set(lineage.lineage_id, lineageColor(lineage.lineage_id, index));
    });
    playback.branches.forEach((branch) => {
      const lineage = this.lineageColors.get(branch.lineage_id);
      this.branchColors.set(branch.branch_id, branchColor(lineage, branch.branch_id));
      this.branchLineages.set(branch.branch_id, branch.lineage_id);
      if (branch.parent_branch_id) this.branchParents.set(branch.branch_id, branch.parent_branch_id);
    });
  }

  resize(width, height) {
    const oldW = this.width;
    const oldH = this.height;
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    if (oldW > 1 && oldH > 1) {
      const sx = this.width / oldW;
      const sy = this.height / oldH;
      for (const particle of this.particles.values()) {
        particle.x *= sx;
        particle.y *= sy;
      }
      for (const particle of this.deadParticles) {
        particle.x *= sx;
        particle.y *= sy;
      }
      for (const body of this.lineageBodies.values()) {
        body.x *= sx;
        body.y *= sy;
        body.vx *= sx;
        body.vy *= sy;
      }
      for (const body of this.branchBodies.values()) {
        body.x *= sx;
        body.y *= sy;
        body.vx *= sx;
        body.vy *= sy;
      }
    }
  }

  setOptions(next) {
    Object.assign(this.options, next);
  }

  rebuild(state) {
    this.particles.clear();
    this.deadParticles = [];
    this.lastTick = state.tick;
    this.stepIndex = state.tick * 12;
    this.syncBodiesToState(state, true);
    for (let i = 0; i < 90; i += 1) {
      this.stepBodies(16.667);
    }
    for (const agent of state.living) {
      this.particles.set(agent.agent_id, this.createParticle(agent, state.tick, true));
    }
    this.recomputeBranchCenters();
  }

  syncToState(state) {
    this.syncBodiesToState(state, false);
    const livingIds = new Set(state.living.map((agent) => agent.agent_id));
    for (const [id, particle] of this.particles) {
      if (!livingIds.has(id)) {
        particle.deathAge = 0;
        if (this.options.showDying) this.deadParticles.push(particle);
        this.particles.delete(id);
      }
    }
    for (const agent of state.living) {
      if (!this.particles.has(agent.agent_id)) {
        const parent = (agent.parent_agent_ids || []).map((id) => this.particles.get(id)).find(Boolean);
        this.particles.set(agent.agent_id, this.createParticle(agent, state.tick, false, parent));
      }
    }
    this.lastTick = state.tick;
  }

  syncBodiesToState(state, reset) {
    const activeLineages = new Set(state.lineageCounts.keys());
    const activeBranches = new Set(state.branchCounts.keys());

    if (reset) {
      this.lineageBodies.clear();
      this.branchBodies.clear();
    }

    for (const [lineageId, count] of state.lineageCounts) {
      if (!this.lineageBodies.has(lineageId)) this.lineageBodies.set(lineageId, this.createLineageBody(lineageId));
      const body = this.lineageBodies.get(lineageId);
      body.count = count;
      body.targetRadius = this.lineageRadius(count);
      body.fade = 1;
      body.extinctAge = null;
    }
    for (const [lineageId, body] of this.lineageBodies) {
      if (!activeLineages.has(lineageId)) {
        body.count = 0;
        body.targetRadius = Math.max(8, body.targetRadius * 0.92);
        if (body.extinctAge === null) body.extinctAge = 0;
      }
    }

    for (const [branchId, count] of state.branchCounts) {
      if (!this.branchBodies.has(branchId)) this.branchBodies.set(branchId, this.createBranchBody(branchId));
      const body = this.branchBodies.get(branchId);
      body.count = count;
      body.targetRadius = this.branchRadius(count);
      body.fade = 1;
      body.extinctAge = null;
      body.lineageId = this.branchLineages.get(branchId);
    }
    for (const [branchId, body] of this.branchBodies) {
      if (!activeBranches.has(branchId)) {
        body.count = 0;
        body.targetRadius = Math.max(6, body.targetRadius * 0.9);
        if (body.extinctAge === null) body.extinctAge = 0;
      }
    }
  }

  createLineageBody(lineageId) {
    const rand = mulberry32(hashString(`${this.seed}:lineage:${lineageId}`));
    const index = this.lineageIndex.get(lineageId) || 0;
    const count = Math.max(1, this.lineageIndex.size);
    const golden = Math.PI * (3 - Math.sqrt(5));
    const ring = Math.min(this.width, this.height) * (0.14 + 0.26 * Math.sqrt((index + 1) / count));
    const angle = index * golden + rand() * 0.36;
    return {
      id: lineageId,
      x: this.width * 0.5 + Math.cos(angle) * ring,
      y: this.height * 0.52 + Math.sin(angle) * ring * 0.72,
      vx: (rand() - 0.5) * 0.18,
      vy: (rand() - 0.5) * 0.18,
      count: 0,
      targetRadius: 28,
      fade: 1,
      extinctAge: null,
    };
  }

  createBranchBody(branchId) {
    const lineageId = this.branchLineages.get(branchId);
    const lineage = this.lineageBodies.get(lineageId) || this.createLineageBody(lineageId);
    const rand = mulberry32(hashString(`${this.seed}:branch:${branchId}`));
    const angle = rand() * Math.PI * 2;
    const distance = 8 + rand() * 18;
    return {
      id: branchId,
      lineageId,
      x: lineage.x + Math.cos(angle) * distance,
      y: lineage.y + Math.sin(angle) * distance,
      vx: (rand() - 0.5) * 0.12,
      vy: (rand() - 0.5) * 0.12,
      count: 0,
      targetRadius: 14,
      fade: 1,
      extinctAge: null,
    };
  }

  lineageRadius(count) {
    return clamp(20 + Math.sqrt(Math.max(0, count)) * 3.05, 22, Math.min(this.width, this.height) * 0.18);
  }

  branchRadius(count) {
    return clamp(9 + Math.sqrt(Math.max(0, count)) * 2.55, 10, Math.min(this.width, this.height) * 0.14);
  }

  createParticle(agent, tick, settled = false, parent = null) {
    const rand = mulberry32(hashString(`${this.seed}:${agent.agent_id}`));
    const branch = this.branchBodies.get(agent.branch_id) || this.createBranchBody(agent.branch_id);
    const angle = rand() * Math.PI * 2;
    const radius = settled ? branch.targetRadius * (0.18 + rand() * 0.58) : 7 + rand() * 9;
    return {
      id: agent.agent_id,
      agent,
      lineageId: agent.lineage_id,
      branchId: agent.branch_id,
      x: parent ? parent.x + Math.cos(angle) * radius : branch.x + Math.cos(angle) * radius,
      y: parent ? parent.y + Math.sin(angle) * radius : branch.y + Math.sin(angle) * radius,
      vx: (rand() - 0.5) * 0.1,
      vy: (rand() - 0.5) * 0.1,
      age: settled ? 2 : 0,
      deathAge: null,
      baseRadius: 2.25 + rand() * 1.48,
      color: this.branchColors.get(agent.branch_id),
    };
  }

  step(dt, state) {
    const start = performance.now();
    this.syncBodiesToState(state, false);
    this.stepBodies(dt);
    this.stepParticles(dt);
    this.stepIndex += 1;
    return performance.now() - start;
  }

  stepBodies(dt) {
    const dtScale = Math.min(1.4, dt / 16.667);
    const motion = Math.max(0.25, this.options.motion);
    const centerX = this.width * 0.5;
    const centerY = this.height * 0.52;
    const lineageBodies = Array.from(this.lineageBodies.values());
    const crowdedLineages = lineageBodies.length > 8;

    for (const body of lineageBodies) {
      let fx = (centerX - body.x) * (crowdedLineages ? 0.00018 : 0.00036);
      let fy = (centerY - body.y) * (crowdedLineages ? 0.00018 : 0.00036);
      for (const other of lineageBodies) {
        if (other === body) continue;
        const { d, nx, ny } = norm(body.x - other.x, body.y - other.y);
        const target = body.targetRadius + other.targetRadius + (crowdedLineages ? 48 : 34);
        if (d < target) {
          const push = (target - d) / target * (crowdedLineages ? 0.08 : 0.068);
          fx += nx * push;
          fy += ny * push;
        } else if (d > target * 3.4) {
          const pull = Math.min(0.004, (d - target * 3.4) * 0.000002);
          fx -= nx * pull;
          fy -= ny * pull;
        }
      }
      const margin = body.targetRadius + 26;
      if (body.x < margin) fx += (margin - body.x) * 0.0026;
      if (body.x > this.width - margin) fx -= (body.x - (this.width - margin)) * 0.0026;
      if (body.y < margin) fy += (margin - body.y) * 0.0026;
      if (body.y > this.height - margin) fy -= (body.y - (this.height - margin)) * 0.0026;
      body.vx = (body.vx + fx * dtScale * motion) * 0.9;
      body.vy = (body.vy + fy * dtScale * motion) * 0.9;
      const maxV = (crowdedLineages ? 0.9 : 1.05) * motion;
      body.vx = clamp(body.vx, -maxV, maxV);
      body.vy = clamp(body.vy, -maxV, maxV);
      body.x += body.vx * dtScale;
      body.y += body.vy * dtScale;
      if (body.extinctAge !== null) {
        body.extinctAge += dt / 1000;
        body.fade = clamp(1 - body.extinctAge / 1.2, 0, 1);
      }
    }

    for (const body of this.branchBodies.values()) {
      const lineage = this.lineageBodies.get(body.lineageId);
      if (!lineage) continue;
      let fx = (lineage.x - body.x) * 0.0021 * this.options.lineagePull;
      let fy = (lineage.y - body.y) * 0.0021 * this.options.lineagePull;
      const siblings = Array.from(this.branchBodies.values()).filter((candidate) => candidate !== body && candidate.lineageId === body.lineageId);
      for (const other of siblings) {
        const { d, nx, ny } = norm(body.x - other.x, body.y - other.y);
        const target = (body.targetRadius + other.targetRadius) * 0.78 + 9;
        if (d < target) {
          const push = (target - d) / target * 0.052 * this.options.branchPull;
          fx += nx * push;
          fy += ny * push;
        }
      }
      const parent = this.branchBodies.get(this.branchParents.get(body.id));
      if (parent) {
        fx += (parent.x - body.x) * 0.00055;
        fy += (parent.y - body.y) * 0.00055;
      }
      const { d, nx, ny } = norm(body.x - lineage.x, body.y - lineage.y);
      const softLimit = Math.max(18, lineage.targetRadius * 0.72);
      if (d > softLimit) {
        const pull = (d - softLimit) * 0.0009;
        fx -= nx * pull;
        fy -= ny * pull;
      }
      body.vx = (body.vx + fx * dtScale * motion) * 0.88;
      body.vy = (body.vy + fy * dtScale * motion) * 0.88;
      const maxV = 1.1 * motion;
      body.vx = clamp(body.vx, -maxV, maxV);
      body.vy = clamp(body.vy, -maxV, maxV);
      body.x += body.vx * dtScale;
      body.y += body.vy * dtScale;
      if (body.extinctAge !== null) {
        body.extinctAge += dt / 1000;
        body.fade = clamp(1 - body.extinctAge / 0.95, 0, 1);
      }
    }
    for (const [lineageId, body] of this.lineageBodies) {
      if (body.fade <= 0) this.lineageBodies.delete(lineageId);
    }
    for (const [branchId, body] of this.branchBodies) {
      if (body.fade <= 0) this.branchBodies.delete(branchId);
    }
  }

  stepParticles(dt) {
    const particles = Array.from(this.particles.values());
    this.recomputeBranchCenters(particles);
    const denseMode = particles.length > 2600;
    const cellSize = denseMode ? 19 : 16;
    const repelDistance2 = denseMode ? 169 : 196;
    const grid = new Map();
    if (!denseMode) {
      for (const particle of particles) {
        const gx = Math.floor(particle.x / cellSize);
        const gy = Math.floor(particle.y / cellSize);
        const key = `${gx}:${gy}`;
        if (!grid.has(key)) grid.set(key, []);
        grid.get(key).push(particle);
      }
    }

    const motion = this.options.motion;
    const dtScale = Math.min(1.4, dt / 16.667);
    for (const particle of particles) {
      const branch = this.branchBodies.get(particle.branchId);
      const lineage = this.lineageBodies.get(particle.lineageId);
      let fx = 0;
      let fy = 0;
      if (branch) {
        const pull = denseMode ? 0.00038 : 0.0028;
        fx += (branch.x - particle.x) * pull * this.options.branchPull;
        fy += (branch.y - particle.y) * pull * this.options.branchPull;
      }
      if (lineage) {
        const pull = denseMode ? 0.00005 : 0.00042;
        fx += (lineage.x - particle.x) * pull * this.options.lineagePull;
        fy += (lineage.y - particle.y) * pull * this.options.lineagePull;
      }

      if (!denseMode) {
        const gx = Math.floor(particle.x / cellSize);
        const gy = Math.floor(particle.y / cellSize);
        for (let oy = -1; oy <= 1; oy += 1) {
          for (let ox = -1; ox <= 1; ox += 1) {
            const bucket = grid.get(`${gx + ox}:${gy + oy}`) || [];
            for (const other of bucket) {
              if (other === particle) continue;
              const dx = particle.x - other.x;
              const dy = particle.y - other.y;
              const d2 = dx * dx + dy * dy;
              if (d2 > 0.001 && d2 < repelDistance2) {
                const force = (repelDistance2 - d2) / repelDistance2 * 0.017;
                const inv = 1 / Math.sqrt(d2);
                fx += dx * inv * force;
                fy += dy * inv * force;
              }
            }
          }
        }
      }

      const driftRand = mulberry32(hashString(`${particle.id}:${Math.floor(this.stepIndex / 18)}`));
      fx += (driftRand() - 0.5) * 0.0015;
      fy += (driftRand() - 0.5) * 0.0015;
      particle.vx = (particle.vx + fx * dtScale * motion) * 0.93;
      particle.vy = (particle.vy + fy * dtScale * motion) * 0.93;
      const maxV = 1.45 * Math.max(0.25, motion);
      particle.vx = clamp(particle.vx, -maxV, maxV);
      particle.vy = clamp(particle.vy, -maxV, maxV);
      particle.x = clamp(particle.x + particle.vx * dtScale, 20, this.width - 20);
      particle.y = clamp(particle.y + particle.vy * dtScale, 20, this.height - 20);
      particle.age += dt / 1000;
    }

    for (const particle of this.deadParticles) {
      particle.deathAge += dt / 1000;
      particle.x += particle.vx * 0.35 * dtScale;
      particle.y += particle.vy * 0.35 * dtScale;
    }
    this.deadParticles = this.deadParticles.filter((particle) => particle.deathAge < 0.85);
  }

  recomputeBranchCenters(particles = Array.from(this.particles.values())) {
    this.branchCenters.clear();
    const branchAcc = new Map();
    for (const particle of particles) {
      const acc = branchAcc.get(particle.branchId) || { x: 0, y: 0, n: 0 };
      acc.x += particle.x;
      acc.y += particle.y;
      acc.n += 1;
      branchAcc.set(particle.branchId, acc);
    }
    for (const [branchId, acc] of branchAcc) {
      this.branchCenters.set(branchId, { x: acc.x / acc.n, y: acc.y / acc.n, n: acc.n });
    }
  }

  snapshot(limit = 48) {
    const particleRows = Array.from(this.particles.values())
      .slice(0, limit)
      .map((p) => ({
        id: p.id,
        lineageId: p.lineageId,
        branchId: p.branchId,
        x: Number(p.x.toFixed(3)),
        y: Number(p.y.toFixed(3)),
        vx: Number(p.vx.toFixed(4)),
        vy: Number(p.vy.toFixed(4)),
        radius: Number(this.visualRadius(p).toFixed(3)),
        phase: Number(p.age.toFixed(3)),
        state: "alive",
      }));
    const dyingRows = this.deadParticles.slice(0, limit).map((p) => ({
      id: p.id,
      lineageId: p.lineageId,
      branchId: p.branchId,
      x: Number(p.x.toFixed(3)),
      y: Number(p.y.toFixed(3)),
      vx: Number(p.vx.toFixed(4)),
      vy: Number(p.vy.toFixed(4)),
      radius: Number(this.visualRadius(p).toFixed(3)),
      phase: Number((p.deathAge || 0).toFixed(3)),
      state: "dying",
    }));
    return {
      particles: [...particleRows, ...dyingRows].sort((a, b) => a.id.localeCompare(b.id)),
      lineageBodies: Array.from(this.lineageBodies.values()).map((b) => ({
        id: b.id,
        count: b.count,
        x: Number(b.x.toFixed(3)),
        y: Number(b.y.toFixed(3)),
        vx: Number(b.vx.toFixed(4)),
        vy: Number(b.vy.toFixed(4)),
        radius: Number(b.targetRadius.toFixed(3)),
        fade: Number(b.fade.toFixed(3)),
      })).sort((a, b) => a.id.localeCompare(b.id)),
      branchBodies: Array.from(this.branchBodies.values()).map((b) => ({
        id: b.id,
        lineageId: b.lineageId,
        count: b.count,
        x: Number(b.x.toFixed(3)),
        y: Number(b.y.toFixed(3)),
        vx: Number(b.vx.toFixed(4)),
        vy: Number(b.vy.toFixed(4)),
        radius: Number(b.targetRadius.toFixed(3)),
        fade: Number(b.fade.toFixed(3)),
      })).sort((a, b) => a.id.localeCompare(b.id)),
    };
  }

  visualRadius(particle) {
    const birth = this.options.showBirthPulse ? 1 + (1 - easeOutCubic(particle.age / 0.58)) * 1.05 : 1;
    return particle.baseRadius * this.options.particleScale * birth;
  }
}
