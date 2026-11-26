// Landmark guard: cleans MediaPipe output before the rep engine sees it.
//
// Three failure modes of a single-frame pose model that show up as false or missed reps:
//
// 1. Left/right identity swaps. In side views the network sometimes labels the near leg as the far
//    leg for a few frames. The joint angle then jumps and the FSM can see a fake "bottom".
//    Fix: compare the frame with the previous output under the identity and the swapped labelling
//    and keep the cheaper one (a two-hypothesis assignment, like a tracker's data association).
// 2. Implausible joints (incorrect pose). A joint that moves faster than a human limb can
//    (here 8 torso lengths per second) is a detection error, not motion. It is marked invisible.
//    If many joints jump at once, the whole frame is rejected (e.g. the detector jumped to
//    another person).
// 3. Short occlusions. A joint hidden for less than `maxGapS` (a passer-by, a dumbbell in front of
//    the knee) is held at its last trusted position with a reduced visibility, so the rep in
//    progress is not lost. Longer gaps are left missing: guessing for longer would invent motion.
//
// The guard works online (no look-ahead), so it adds no latency.

import { LM, type Point } from "./angles";

export type GuardOptions = {
  fixSwaps: boolean;
  rejectImplausible: boolean;
  fillGaps: boolean;
  maxGapS: number;        // longest occlusion that is bridged
  maxSpeed: number;       // torso lengths per second
  filledVisibility: number;
};

export const DEFAULT_GUARD: GuardOptions = {
  fixSwaps: true, rejectImplausible: true, fillGaps: true, maxGapS: 0.6, maxSpeed: 8, filledVisibility: 0.55,
};

export type GuardStats = { frames: number; swapsFixed: number; jointsRejected: number; framesRejected: number; jointsFilled: number };

const PAIRS: [number, number][] = [
  [LM.lShoulder, LM.rShoulder], [LM.lElbow, LM.rElbow], [LM.lWrist, LM.rWrist], [LM.lHip, LM.rHip],
  [LM.lKnee, LM.rKnee], [LM.lAnkle, LM.rAnkle], [LM.lHeel, LM.rHeel], [LM.lFoot, LM.rFoot],
];
// every left/right pair of the 33-point topology (face and hands included) for the swap itself
const ALL_PAIRS: [number, number][] = [
  [1, 4], [2, 5], [3, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22],
  [23, 24], [25, 26], [27, 28], [29, 30], [31, 32],
];
const BODY = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
const VIS_OK = 0.5;

type Track = { x: number; y: number; t: number };

const d2 = (a: Point, b: Point, aspect: number) => Math.hypot((a.x - b.x) * aspect, a.y - b.y);

export function torsoLength(lms: Point[], aspect = 1): number {
  const sx = (lms[LM.lShoulder].x + lms[LM.rShoulder].x) / 2, sy = (lms[LM.lShoulder].y + lms[LM.rShoulder].y) / 2;
  const hx = (lms[LM.lHip].x + lms[LM.rHip].x) / 2, hy = (lms[LM.lHip].y + lms[LM.rHip].y) / 2;
  return Math.hypot((sx - hx) * aspect, sy - hy);
}

export function swapSides(lms: Point[]): Point[] {
  const out = lms.slice();
  for (const [l, r] of ALL_PAIRS) { out[l] = lms[r]; out[r] = lms[l]; }
  return out;
}

export class LandmarkGuard {
  readonly opts: GuardOptions;
  stats: GuardStats = { frames: 0, swapsFixed: 0, jointsRejected: 0, framesRejected: 0, jointsFilled: 0 };
  private prev: Point[] | null = null;
  private prevT = 0;
  private good: (Track | null)[] = new Array(33).fill(null);
  private torso = 0;

  constructor(opts: Partial<GuardOptions> = {}) {
    this.opts = { ...DEFAULT_GUARD, ...opts };
  }

  /** Returns cleaned landmarks (same normalised coordinates) or null when nothing usable is left. */
  process(lms: Point[] | null | undefined, t: number, aspect = 1): Point[] | null {
    this.stats.frames += 1;
    const o = this.opts;
    if (!lms || lms.length < 33) return this.fillAll(t);
    let cur = lms.map((p) => ({ ...p }));

    const tl = torsoLength(cur, aspect);
    if (tl > 1e-3) this.torso = this.torso ? 0.9 * this.torso + 0.1 * tl : tl;

    if (o.fixSwaps && this.prev && t - this.prevT < 0.5) {
      let same = 0, swapped = 0;
      for (const [l, r] of PAIRS) {
        same += d2(cur[l], this.prev[l], aspect) + d2(cur[r], this.prev[r], aspect);
        swapped += d2(cur[r], this.prev[l], aspect) + d2(cur[l], this.prev[r], aspect);
      }
      if (swapped < 0.6 * same && same > 0.25 * this.torso) {
        cur = swapSides(cur);
        this.stats.swapsFixed += 1;
      }
    }

    if (o.rejectImplausible && this.torso > 0) {
      let jumps = 0;
      for (const i of BODY) {
        const g = this.good[i];
        if (!g || (cur[i].visibility ?? 0) < VIS_OK) continue;
        const dt = Math.max(1 / 60, t - g.t);
        const speed = Math.hypot((cur[i].x - g.x) * aspect, cur[i].y - g.y) / this.torso / dt;
        if (speed > o.maxSpeed && dt < 0.5) {
          cur[i] = { ...cur[i], visibility: 0 };
          jumps += 1;
        }
      }
      this.stats.jointsRejected += jumps;
      if (jumps >= 5) {
        this.stats.framesRejected += 1;
        return this.fillAll(t);
      }
    }

    if (o.fillGaps) {
      for (const i of BODY) {
        if ((cur[i].visibility ?? 0) >= VIS_OK) continue;
        const g = this.good[i];
        if (g && t - g.t <= o.maxGapS) {
          cur[i] = { ...cur[i], x: g.x, y: g.y, visibility: o.filledVisibility };
          this.stats.jointsFilled += 1;
        }
      }
    }

    for (const i of BODY) {
      const p = cur[i];
      if ((p.visibility ?? 0) >= VIS_OK && p.visibility !== o.filledVisibility) this.good[i] = { x: p.x, y: p.y, t };
    }
    this.prev = cur;
    this.prevT = t;
    return cur;
  }

  /** Whole frame missing: bridge a short dropout from the last trusted positions. */
  private fillAll(t: number): Point[] | null {
    const o = this.opts;
    if (!o.fillGaps || !this.prev) return null;
    const out = this.prev.map((p) => ({ ...p, visibility: Math.min(p.visibility ?? 0, 0.3) }));
    let n = 0;
    for (const i of BODY) {
      const g = this.good[i];
      if (g && t - g.t <= o.maxGapS) {
        out[i] = { x: g.x, y: g.y, visibility: o.filledVisibility };
        n += 1;
      }
    }
    if (n < 8) return null;
    this.stats.jointsFilled += n;
    return out;
  }
}
