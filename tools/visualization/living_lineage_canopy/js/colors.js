export function hashString(value) {
  let h = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function rand() {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function lineageColor(lineageId, index = 0) {
  const base = (index * 137.508 + (hashString(lineageId) % 31)) % 360;
  return { h: base, s: 58, l: 62 };
}

export function branchColor(lineage, branchId) {
  const shift = ((hashString(branchId) % 41) - 20) * 0.65;
  return {
    h: (lineage.h + shift + 360) % 360,
    s: Math.max(42, Math.min(66, lineage.s + ((hashString(branchId) % 9) - 4))),
    l: Math.max(52, Math.min(70, lineage.l + ((hashString(branchId) % 7) - 3))),
  };
}

export function hsla(color, alpha = 1, lightnessOffset = 0) {
  return `hsla(${color.h.toFixed(1)}, ${color.s}%, ${Math.max(0, Math.min(100, color.l + lightnessOffset))}%, ${alpha})`;
}
