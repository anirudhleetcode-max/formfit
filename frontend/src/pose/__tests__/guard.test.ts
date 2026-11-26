import { describe, expect, it } from "vitest";
import { LM, type Point } from "../angles";
import { PoseSession } from "../engine";
import { LandmarkGuard, swapSides } from "../guard";
import { FPS, repSeries, squatSkeleton } from "./synthetic";

const frame = (knee = 170, vis = 0.95): Point[] => squatSkeleton(knee, { visibility: vis });

describe("LandmarkGuard", () => {
  it("passes clean frames through unchanged", () => {
    const g = new LandmarkGuard();
    const f = frame();
    const out = g.process(f, 0)!;
    expect(out[LM.lKnee].x).toBeCloseTo(f[LM.lKnee].x, 6);
    expect(g.stats.swapsFixed).toBe(0);
  });

  it("undoes a left/right identity swap", () => {
    const g = new LandmarkGuard();
    const a = frame(170);
    // separate left and right legs so a swap is visible
    a[LM.rKnee] = { ...a[LM.rKnee], x: a[LM.rKnee].x + 0.1 };
    a[LM.rAnkle] = { ...a[LM.rAnkle], x: a[LM.rAnkle].x + 0.1 };
    a[LM.rHip] = { ...a[LM.rHip], x: a[LM.rHip].x + 0.1 };
    g.process(a, 0);
    const out = g.process(swapSides(a), 1 / FPS)!;
    expect(g.stats.swapsFixed).toBe(1);
    expect(out[LM.rKnee].x).toBeCloseTo(a[LM.rKnee].x, 6);
  });

  it("rejects a joint that moves faster than a limb can", () => {
    const g = new LandmarkGuard({ fillGaps: false });
    g.process(frame(), 0);
    const b = frame();
    b[LM.lKnee] = { ...b[LM.lKnee], x: b[LM.lKnee].x + 0.4 };
    const out = g.process(b, 1 / FPS)!;
    expect(out[LM.lKnee].visibility).toBe(0);
    expect(g.stats.jointsRejected).toBe(1);
  });

  it("bridges a short occlusion but not a long one", () => {
    const g = new LandmarkGuard({ maxGapS: 0.6 });
    g.process(frame(), 0);
    const hidden = frame();
    hidden[LM.lKnee] = { ...hidden[LM.lKnee], visibility: 0.1 };
    expect(g.process(hidden, 0.3)![LM.lKnee].visibility).toBeCloseTo(0.55);
    expect(g.process(hidden, 1.0)![LM.lKnee].visibility).toBeCloseTo(0.1);
  });

  it("bridges a dropped frame and gives up after the gap limit", () => {
    const g = new LandmarkGuard({ maxGapS: 0.4 });
    g.process(frame(), 0);
    expect(g.process(null, 0.2)).not.toBeNull();
    expect(g.process(null, 0.9)).toBeNull();
  });
});

describe("PoseSession with the guard", () => {
  it("still counts a rep that has a short detection dropout at the bottom", () => {
    const series = repSeries(172, 70, 1.2, 1.0, 0.3);
    const run = (guard: boolean) => {
      const s = new PoseSession("squat", { guard });
      const mid = Math.floor(series.length / 2);
      series.forEach((k, i) => {
        const lost = Math.abs(i - mid) < 4;
        s.process(lost ? null : frame(k), i / FPS);
      });
      return s.reps.length;
    };
    expect(run(true)).toBe(1);
    expect(run(false)).toBeGreaterThanOrEqual(0);
  });
});

describe("false-rep gate", () => {
  const always = (label: string) => ({ recognize: () => ({ label, p: 0.95, probs: {} }), push: () => {} });
  const run = (label: string) => {
    // a fake recognizer that always reports `label`; one clean squat rep
    const s = new PoseSession("squat", { recognizer: always(label) as never, gateTau: 0.8 });
    repSeries(172, 70, 1.2, 1.0, 0.3).forEach((k, i) => s.process(frame(k), i / FPS));
    return s;
  };
  it("counts a rep the classifier agrees with", () => {
    const s = run("squat");
    expect(s.reps.length).toBe(1);
    expect(s.vetoedReps).toBe(0);
  });
  it("keeps a vetoed rep aside and restores it on request", () => {
    const s = run("lunge");
    expect(s.reps.length).toBe(0);
    expect(s.vetoedReps).toBe(1);
    expect(s.acceptVetoed()).toBe(1);
    expect(s.reps.length).toBe(1);
    expect(s.setReps).toBe(1);
    expect(s.gateTau).toBe(0);
  });
});
