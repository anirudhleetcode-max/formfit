// Geometry helpers and signal filters. Pure functions, no DOM.

export type Point = { x: number; y: number; z?: number; visibility?: number };

/** MediaPipe Pose landmark indices (33-point BlazePose topology). */
export const LM = {
  nose: 0,
  lShoulder: 11, rShoulder: 12,
  lElbow: 13, rElbow: 14,
  lWrist: 15, rWrist: 16,
  lHip: 23, rHip: 24,
  lKnee: 25, rKnee: 26,
  lAnkle: 27, rAnkle: 28,
  lHeel: 29, rHeel: 30,
  lFoot: 31, rFoot: 32,
} as const;

/** Skeleton edges drawn on the overlay. */
export const EDGES: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],
  [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
];

export const DEG = 180 / Math.PI;

export function dist(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Interior angle ABC in degrees (0..180), 2-D image plane. */
export function angleDeg(a: Point, b: Point, c: Point): number {
  const v1x = a.x - b.x, v1y = a.y - b.y;
  const v2x = c.x - b.x, v2y = c.y - b.y;
  const n = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y);
  if (n === 0) return NaN;
  const cos = Math.min(1, Math.max(-1, (v1x * v2x + v1y * v2y) / n));
  return Math.acos(cos) * DEG;
}

/** Unsigned angle of the segment from->to measured from the vertical (0 = upright). */
export function angleFromVertical(from: Point, to: Point): number {
  const dx = to.x - from.x, dy = from.y - to.y; // image y grows downwards
  if (dx === 0 && dy === 0) return NaN;
  return Math.abs(Math.atan2(dx, dy)) * DEG;
}

/** Signed perpendicular offset of p from the line a->c, normalised by |ac|.
 *  Positive when p lies below the line in image coordinates (e.g. sagging hips). */
export function lineOffset(a: Point, c: Point, p: Point): number {
  const dx = c.x - a.x, dy = c.y - a.y;
  const len = Math.hypot(dx, dy);
  if (len === 0) return 0;
  const cross = (dx * (p.y - a.y) - dy * (p.x - a.x)) / len;
  // make the sign independent of which way the body faces
  return (dx >= 0 ? cross : -cross) / len;
}

export function mid(a: Point, b: Point): Point {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, visibility: Math.min(a.visibility ?? 1, b.visibility ?? 1) };
}

export function vis(p: Point | undefined): number {
  return p?.visibility ?? 0;
}

export function minVisibility(lms: Point[], idx: readonly number[]): number {
  let m = 1;
  for (const i of idx) m = Math.min(m, vis(lms[i]));
  return m;
}

/** Pick the body side the camera sees best, given left/right landmark index lists. */
export function bestSide(lms: Point[], left: readonly number[], right: readonly number[]): "left" | "right" {
  const s = (idx: readonly number[]) => idx.reduce((acc, i) => acc + vis(lms[i]), 0);
  return s(left) >= s(right) ? "left" : "right";
}

/** Exponential moving average. */
export class Ema {
  private v: number | null = null;
  private alpha: number;
  constructor(alpha: number) { this.alpha = alpha; }
  next(x: number): number {
    if (!Number.isFinite(x)) return this.v ?? NaN;
    this.v = this.v === null ? x : this.alpha * x + (1 - this.alpha) * this.v;
    return this.v;
  }
  reset() { this.v = null; }
}

/** One Euro filter (Casiez et al., CHI 2012): adaptive low-pass — smooth when still,
 *  responsive when moving. t is in seconds. */
export class OneEuro {
  private xPrev: number | null = null;
  private dxPrev = 0;
  private tPrev = 0;
  private minCutoff: number;
  private beta: number;
  private dCutoff: number;
  constructor(minCutoff = 1.0, beta = 0.02, dCutoff = 1.0) {
    this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
  }

  private static alpha(cutoff: number, dt: number) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  next(x: number, t: number): number {
    if (!Number.isFinite(x)) return this.xPrev ?? NaN;
    if (this.xPrev === null) {
      this.xPrev = x; this.tPrev = t; this.dxPrev = 0;
      return x;
    }
    const dt = Math.max(1e-3, t - this.tPrev);
    const dx = (x - this.xPrev) / dt;
    const aD = OneEuro.alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = OneEuro.alpha(cutoff, dt);
    const xHat = a * x + (1 - a) * this.xPrev;
    this.xPrev = xHat; this.dxPrev = dxHat; this.tPrev = t;
    return xHat;
  }

  reset() { this.xPrev = null; this.dxPrev = 0; }
}
