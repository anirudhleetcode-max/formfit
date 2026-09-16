// Synthetic signals and skeletons for tests.
import type { Point } from "../angles";

export const FPS = 30;

/** Cosine-eased rep: hold top, go down to `bottom`, pause, come back up, hold. */
export function repSeries(top: number, bottom: number, downS: number, upS: number, pauseS = 0.2, holdS = 0.4): number[] {
  const out: number[] = [];
  const push = (n: number, f: (u: number) => number) => {
    for (let i = 0; i < n; i++) out.push(f(n === 1 ? 1 : i / (n - 1)));
  };
  push(Math.round(holdS * FPS), () => top);
  push(Math.round(downS * FPS), (u) => top + (bottom - top) * (1 - Math.cos(Math.PI * u)) / 2);
  push(Math.round(pauseS * FPS), () => bottom);
  push(Math.round(upS * FPS), (u) => bottom + (top - bottom) * (1 - Math.cos(Math.PI * u)) / 2);
  push(Math.round(holdS * FPS), () => top);
  return out;
}

/** Deterministic pseudo-random noise (mulberry32). */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gauss(r: () => number) {
  return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
}

const R = Math.PI / 180;

/** Side-view squat skeleton (normalised image coords) for a given knee angle.
 *  extraLean adds forward torso lean on top of the balance-driven lean. */
export function squatSkeleton(kneeAngle: number, opts: { extraLean?: number; heelLift?: number; visibility?: number } = {}): Point[] {
  const flex = 180 - kneeAngle;
  const shank = Math.min(35, 0.5 * flex) * R;
  const thigh = flex * R - shank;
  const lean = (0.3 * flex + (opts.extraLean ?? 0)) * R;
  const Ls = 0.2, Lt = 0.2, Lk = 0.25;
  const v = opts.visibility ?? 0.95;
  const ankle = { x: 0.5, y: 0.88 };
  const knee = { x: ankle.x + Ls * Math.sin(shank), y: ankle.y - Ls * Math.cos(shank) };
  const hip = { x: knee.x - Lt * Math.sin(thigh), y: knee.y - Lt * Math.cos(thigh) };
  const sh = { x: hip.x + Lk * Math.sin(lean), y: hip.y - Lk * Math.cos(lean) };
  const elbow = { x: sh.x + 0.1, y: sh.y + 0.03 };
  const wrist = { x: elbow.x + 0.1, y: elbow.y };
  const lift = opts.heelLift ?? 0;
  const heel = { x: ankle.x - 0.03, y: 0.93 - lift * 0.1 };
  const toe = { x: ankle.x + 0.08, y: 0.93 };
  const nose = { x: sh.x + 0.03, y: sh.y - 0.08 };
  const lms: Point[] = Array.from({ length: 33 }, () => ({ x: nose.x, y: nose.y, visibility: v }));
  const set = (i: number, p: { x: number; y: number }, dx = 0, vis = v) => { lms[i] = { x: p.x + dx, y: p.y, visibility: vis }; };
  // left side faces the camera; right side is behind (lower visibility), shifted slightly
  const pairs: [number, number, { x: number; y: number }][] = [
    [11, 12, sh], [13, 14, elbow], [15, 16, wrist], [23, 24, hip], [25, 26, knee], [27, 28, ankle],
    [29, 30, heel], [31, 32, toe],
  ];
  set(0, nose);
  for (const [l, r, p] of pairs) { set(l, p); set(r, p, 0.01, v * 0.7); }
  return lms;
}

/** Frontal-view squat skeleton with configurable knee-in (valgus) ratio. */
export function frontSquatSkeleton(kneeAngle: number, valgus = 1.1): Point[] {
  const flex = (180 - kneeAngle) * R;
  const hipY = 0.5 + 0.2 * (1 - Math.cos(flex)) * 0.6;
  const kneeY = 0.7 + 0.02 * Math.sin(flex);
  const ankleX = 0.1, kneeX = ankleX * valgus;
  const lms: Point[] = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.15, visibility: 0.95 }));
  const put = (i: number, x: number, y: number) => { lms[i] = { x, y, visibility: 0.95 }; };
  // image left = subject right; only the geometry matters
  put(11, 0.5 + 0.1, hipY - 0.2); put(12, 0.5 - 0.1, hipY - 0.2);
  put(13, 0.62, hipY - 0.1); put(14, 0.38, hipY - 0.1);
  put(15, 0.62, hipY); put(16, 0.38, hipY);
  put(23, 0.5 + 0.06, hipY); put(24, 0.5 - 0.06, hipY);
  // pull the hips forward in projection so the knee angle reads correctly
  const hipDrop = hipY;
  put(25, 0.5 + kneeX, kneeY); put(26, 0.5 - kneeX, kneeY);
  put(27, 0.5 + ankleX, 0.9); put(28, 0.5 - ankleX, 0.9);
  put(29, 0.5 + ankleX, 0.92); put(30, 0.5 - ankleX, 0.92);
  put(31, 0.5 + ankleX, 0.93); put(32, 0.5 - ankleX, 0.93);
  void hipDrop;
  return lms;
}
