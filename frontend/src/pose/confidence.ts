// Tracking confidence: how much we trust the landmarks behind a frame / a rep.
//
// MediaPipe gives each landmark a `visibility` (probability it is visible and not occluded).
// It does not tell us about temporal noise, so we also estimate jitter from the raw joint angle:
// for smooth human motion sampled at >= 10 fps the second difference x[t] - 2x[t-1] + x[t-2]
// is close to zero, while white landmark noise with std s gives |d2| of about 1.95 s on average
// (E|N(0, 6 s^2)| = sqrt(6) * sqrt(2/pi) * s). So mean|d2| / 1.95 estimates the angle noise in degrees.
//
// Frame confidence = visibility score x jitter score (both in [0, 1]).
// Rep confidence   = mean frame confidence x dropout factor x frame-rate factor.
// A rep whose confidence is below REP_MIN is counted but NOT scored (abstention).

import { LM, type Point } from "./angles";

export const FRAME_MIN = 0.4;   // below this: no live cues, no red joints
export const REP_MIN = 0.5;     // below this: rep counted, form not scored
export const MIN_REP_FPS = 8;   // fewer analysed frames per second than this: tempo/angles unreliable

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

/** Maps visibility 0.3 -> 0 and 0.8 -> 1. */
export function visibilityScore(vis: number[]): number {
  if (!vis.length) return 0;
  const min = Math.min(...vis);
  const mean = vis.reduce((a, b) => a + b, 0) / vis.length;
  const m = (v: number) => clamp01((v - 0.3) / 0.5);
  return 0.5 * m(min) + 0.5 * m(mean);
}

/** Noise of 2 deg or less -> 1, 10 deg or more -> 0. */
export function jitterScore(noiseDeg: number): number {
  return clamp01(1 - (noiseDeg - 2) / 8);
}

const GROUPS: [string, number[]][] = [
  ["shoulders", [LM.lShoulder, LM.rShoulder]],
  ["elbows", [LM.lElbow, LM.rElbow]],
  ["wrists", [LM.lWrist, LM.rWrist]],
  ["hips", [LM.lHip, LM.rHip]],
  ["knees", [LM.lKnee, LM.rKnee]],
  ["ankles", [LM.lAnkle, LM.rAnkle]],
];

/** Name of the body part (among the required joints) that is least visible. */
export function weakestPart(lms: Point[] | null | undefined, required: readonly number[]): string {
  if (!lms) return "body";
  let worst = "body", worstV = Infinity;
  for (const [name, idx] of GROUPS) {
    const used = idx.filter((i) => required.includes(i));
    if (!used.length) continue;
    const v = Math.max(...used.map((i) => {
      const p = lms[i];
      const inFrame = p && p.x > -0.02 && p.x < 1.02 && p.y > -0.02 && p.y < 1.02;
      return inFrame ? (p.visibility ?? 0) : 0;
    }));
    if (v < worstV) { worstV = v; worst = name; }
  }
  return worst;
}

/** Running jitter estimate from the raw (unsmoothed) primary angle. */
export class JitterMeter {
  private prev: number[] = [];
  private prevT: number[] = [];
  private ema: number | null = null;
  private alpha: number;
  constructor(alpha = 0.15) { this.alpha = alpha; }

  /** Returns the current noise estimate in degrees (null until enough samples). */
  update(angle: number, t: number): number | null {
    if (!Number.isFinite(angle)) return this.noise;
    this.prev.push(angle);
    this.prevT.push(t);
    if (this.prev.length > 3) { this.prev.shift(); this.prevT.shift(); }
    if (this.prev.length === 3) {
      const gap = this.prevT[2] - this.prevT[0];
      // only meaningful when frames are close together (>= ~10 fps)
      if (gap > 0 && gap <= 0.22) {
        const d2 = Math.abs(this.prev[2] - 2 * this.prev[1] + this.prev[0]);
        this.ema = this.ema === null ? d2 : this.alpha * d2 + (1 - this.alpha) * this.ema;
      }
    }
    return this.noise;
  }

  get noise(): number | null {
    return this.ema === null ? null : this.ema / 1.95;
  }

  reset() { this.prev = []; this.prevT = []; this.ema = null; }
}

export type RepQuality = {
  confidence: number;
  scored: boolean;
  reason: string | null;   // why the rep was not scored
  fps: number;
};

/** Combine per-frame confidences of one rep window into a rep verdict. */
export function repQuality(
  frameConf: number[], framesInWindow: number, durationS: number, weakest: string,
): RepQuality {
  const n = frameConf.length;
  const fps = durationS > 0 ? framesInWindow / durationS : 0;
  if (n === 0) return { confidence: 0, scored: false, reason: `Lost track of your ${weakest}`, fps };
  const mean = frameConf.reduce((a, b) => a + b, 0) / n;
  const dropout = clamp01(n / Math.max(1, framesInWindow));          // share of frames with usable pose
  const fpsFactor = clamp01((fps - 5) / (MIN_REP_FPS + 2 - 5));      // 5 fps -> 0, 10 fps -> 1
  const confidence = Math.round(mean * (0.2 + 0.8 * dropout) * fpsFactor * 100) / 100;
  let reason: string | null = null;
  if (confidence < REP_MIN) {
    if (fps < MIN_REP_FPS) reason = `Too few frames per second (${fps.toFixed(1)}) to judge form`;
    else if (dropout < 0.7) reason = `Lost track of your ${weakest} during the rep`;
    else reason = `Can't see your ${weakest} clearly`;
  }
  return { confidence, scored: confidence >= REP_MIN, reason, fps };
}
