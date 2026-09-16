import { describe, expect, it } from "vitest";
import { JitterMeter, REP_MIN, jitterScore, repQuality, visibilityScore, weakestPart } from "../confidence";
import { PoseSession } from "../engine";
import { FPS, gauss, repSeries, rng, squatSkeleton } from "./synthetic";

describe("confidence primitives", () => {
  it("maps visibility and jitter to [0, 1]", () => {
    expect(visibilityScore([0.95, 0.9])).toBe(1);
    expect(visibilityScore([0.2, 0.25])).toBe(0);
    expect(visibilityScore([])).toBe(0);
    expect(jitterScore(1)).toBe(1);
    expect(jitterScore(6)).toBeCloseTo(0.5);
    expect(jitterScore(15)).toBe(0);
  });

  it("estimates landmark angle noise from the second difference", () => {
    const r = rng(11);
    for (const sigma of [1, 4, 8]) {
      const m = new JitterMeter(0.01);
      let est = 0;
      for (let i = 0; i < 3000; i++) {
        // smooth rep motion + white noise
        const clean = 120 + 50 * Math.cos((2 * Math.PI * i) / (2.5 * FPS));
        est = m.update(clean + sigma * gauss(r), i / FPS) ?? 0;
      }
      expect(est).toBeGreaterThan(sigma * 0.75);
      expect(est).toBeLessThan(sigma * 1.25 + 0.3);
    }
  });

  it("ignores frame pairs that are too far apart", () => {
    const m = new JitterMeter();
    [100, 150, 90, 160].forEach((a, i) => m.update(a, i * 0.5));
    expect(m.noise).toBeNull();
  });

  it("names the least visible required body part", () => {
    const lms = squatSkeleton(170).map((p, i) => (i === 27 ? { ...p, visibility: 0.1 } : p));
    expect(weakestPart(lms, [11, 23, 25, 27])).toBe("ankles");
    expect(weakestPart(null, [11])).toBe("body");
  });

  it("abstains on low confidence, dropouts and low frame rate", () => {
    const good = repQuality(Array(40).fill(0.9), 40, 2, "knees");
    expect(good.scored).toBe(true);
    expect(good.confidence).toBeGreaterThanOrEqual(REP_MIN);
    const blurry = repQuality(Array(40).fill(0.3), 40, 2, "knees");
    expect(blurry).toMatchObject({ scored: false, reason: "Can't see your knees clearly" });
    const dropped = repQuality(Array(8).fill(0.9), 40, 2, "ankles");
    expect(dropped.scored).toBe(false);
    expect(dropped.reason).toMatch(/Lost track of your ankles/);
    const slow = repQuality(Array(6).fill(0.95), 6, 2, "knees");
    expect(slow.scored).toBe(false);
    expect(slow.reason).toMatch(/Too few frames per second/);
    expect(repQuality([], 10, 2, "hips").scored).toBe(false);
  });
});

describe("PoseSession abstention", () => {
  function feed(s: PoseSession, angles: number[], noisePx: number, vis = 0.95, fps = FPS) {
    const r = rng(5);
    return angles.map((a, i) => s.process(
      squatSkeleton(a, { visibility: vis }).map((p) => ({ ...p, x: p.x + noisePx * gauss(r), y: p.y + noisePx * gauss(r) })),
      i / fps,
    ));
  }

  it("scores clean tracking with high confidence", () => {
    const s = new PoseSession("squat");
    const outs = feed(s, repSeries(174, 58, 1.5, 1.0), 0.002);
    expect(s.reps).toHaveLength(1);
    expect(s.reps[0].scored).toBe(true);
    expect(s.reps[0].score).not.toBeNull();
    expect(s.reps[0].confidence).toBeGreaterThan(0.7);
    expect(outs.at(-1)!.confidence).toBeGreaterThan(0.6);
  });

  it("counts but does not score a rep tracked through heavy jitter, and mutes cues", () => {
    const s = new PoseSession("squat");
    const outs = feed(s, repSeries(174, 58, 2.5, 2.0, 0.4, 1.0), 0.02);
    expect(s.reps.length).toBeGreaterThanOrEqual(1);
    const rep = s.reps[0];
    expect(rep.scored).toBe(false);
    expect(rep.score).toBeNull();
    expect(rep.faults).toEqual([]);
    expect(rep.abstainReason).toBeTruthy();
    expect(outs.some((o) => o.lowConfidence)).toBe(true);
    expect(outs.filter((o) => o.lowConfidence).every((o) => o.badJoints.length === 0)).toBe(true);
    expect(s.toApiReps()[0]).toMatchObject({ score: null, scored: false });
  });

  it("does not score reps sampled at a very low frame rate", () => {
    const s = new PoseSession("squat");
    const series = repSeries(174, 58, 1.5, 1.0).filter((_, i) => i % 10 === 0); // 3 fps
    feed(s, series, 0.001, 0.95, 3);
    expect(s.reps).toHaveLength(1);
    expect(s.reps[0].scored).toBe(false);
    expect(s.reps[0].abstainReason).toMatch(/frames per second/);
  });
});
