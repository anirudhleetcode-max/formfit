import { describe, expect, it } from "vitest";
import { CueThrottle } from "../cues";
import { PoseSession } from "../engine";
import { checkVisibility, evaluateRep, frameMetrics, scoreFaults, type RepFeatures } from "../rules";
import { FPS, frontSquatSkeleton, gauss, repSeries, rng, squatSkeleton } from "./synthetic";

const base: RepFeatures = {
  minAngle: 70, maxAngle: 172, rom: 102, eccS: 1.4, conS: 0.9, torsoLeanMax: 30, torsoSway: 25,
  hipLineMin: 175, valgusMin: 1.0, elbowDriftMax: 10, depth: 0.05, heelRise: 0, hipOffsetAtWorst: 0,
};

describe("rep rules", () => {
  it("scores a clean squat 100", () => {
    expect(evaluateRep("squat", base)).toEqual({ score: 100, faults: [] });
  });

  it("flags squat faults with penalties", () => {
    const r = evaluateRep("squat", { ...base, depth: -0.4, torsoLeanMax: 55, valgusMin: 0.7, eccS: 0.3 });
    expect(r.faults).toEqual(["shallow_depth", "forward_lean", "knee_valgus", "rushed_descent"]);
    expect(r.score).toBe(100 - 30 - 20 - 25 - 10);
    expect(scoreFaults(["shallow_depth", "knee_valgus", "hip_sag", "back_arch"])).toBe(0);
  });

  it("separates sagging from piking push-ups", () => {
    const p = { ...base, minAngle: 80, maxAngle: 170, hipLineMin: 150 };
    expect(evaluateRep("pushup", { ...p, hipOffsetAtWorst: 0.05 }).faults).toEqual(["hip_sag"]);
    expect(evaluateRep("pushup", { ...p, hipOffsetAtWorst: -0.05 }).faults).toEqual(["hip_pike"]);
    expect(evaluateRep("pushup", { ...p, minAngle: 110 }).faults).toContain("shallow_pushup");
  });

  it("flags curl swing / drift / range and press lockout / arch", () => {
    const c = { ...base, minAngle: 40, maxAngle: 165, torsoSway: 4, torsoLeanMax: 5 };
    expect(evaluateRep("curl", c).faults).toEqual([]);
    expect(evaluateRep("curl", { ...c, elbowDriftMax: 40, torsoSway: 18 }).faults).toEqual(["elbow_drift", "torso_swing"]);
    expect(evaluateRep("curl", { ...c, minAngle: 85, maxAngle: 140 }).faults).toEqual(["partial_extension", "partial_curl"]);
    const pr = { ...base, minAngle: 70, maxAngle: 170, torsoLeanMax: 5 };
    expect(evaluateRep("press", pr).faults).toEqual([]);
    expect(evaluateRep("press", { ...pr, maxAngle: 150, torsoLeanMax: 20 }).faults).toEqual(["short_press", "back_arch"]);
  });
});

describe("frame metrics", () => {
  it("recovers the knee angle and depth from a side-view skeleton", () => {
    const stand = frameMetrics("squat", squatSkeleton(175), "left");
    const deep = frameMetrics("squat", squatSkeleton(60), "left");
    expect(stand.frontal).toBe(false);
    expect(stand.primary).toBeCloseTo(175, 0);
    expect(deep.primary).toBeCloseTo(60, 0);
    expect(deep.depth).toBeGreaterThan(-0.1);
    expect(stand.depth).toBeLessThan(-0.9);
    expect(deep.torsoLean).toBeGreaterThan(stand.torsoLean);
    expect(frameMetrics("squat", squatSkeleton(80, { heelLift: 0.5 }), "left").heelRise).toBeGreaterThan(0.2);
  });

  it("measures knee valgus only in a frontal view", () => {
    const good = frameMetrics("squat", frontSquatSkeleton(90, 1.1), "left");
    const bad = frameMetrics("squat", frontSquatSkeleton(90, 0.6), "left");
    expect(good.frontal).toBe(true);
    expect(good.valgus).toBeCloseTo(1.1, 1);
    expect(bad.valgus).toBeCloseTo(0.6, 1);
    expect(frameMetrics("squat", squatSkeleton(90), "left").valgus).toBeNull();
  });

  it("asks the user to step back when joints are missing or off-frame", () => {
    expect(checkVisibility("squat", null)).toEqual({ ok: false, reason: "No one in view" });
    expect(checkVisibility("squat", squatSkeleton(170, { visibility: 0.2 })).ok).toBe(false);
    const cut = squatSkeleton(170).map((p, i) => (i === 27 || i === 28 ? { ...p, y: 1.2 } : p));
    expect(checkVisibility("squat", cut)).toEqual({ ok: false, reason: "Step back so your whole body is visible" });
    expect(checkVisibility("squat", squatSkeleton(170))).toEqual({ ok: true, side: "left" });
  });
});

describe("PoseSession (landmarks -> reps)", () => {
  function feed(session: PoseSession, angles: number[], opts: Parameters<typeof squatSkeleton>[1] = {}, seed = 3, t0 = 0) {
    const r = rng(seed);
    const outs = [];
    for (let i = 0; i < angles.length; i++) {
      const lms = squatSkeleton(angles[i], opts).map((p) => ({ ...p, x: p.x + 0.003 * gauss(r), y: p.y + 0.003 * gauss(r) }));
      outs.push(session.process(lms, t0 + i / FPS));
    }
    return outs;
  }

  it("counts noisy squats and scores deep reps higher than shallow ones", () => {
    const s = new PoseSession("squat");
    const deep = repSeries(174, 58, 1.6, 1.0);
    const shallow = repSeries(174, 108, 1.2, 0.9);
    feed(s, [...deep, ...deep, ...deep, ...shallow, ...shallow]);
    expect(s.reps).toHaveLength(5);
    const [d1, , , sh1] = s.reps;
    expect(d1.faults).not.toContain("shallow_depth");
    expect(sh1.faults).toContain("shallow_depth");
    expect(d1.score).toBeGreaterThan(sh1.score);
    expect(d1.eccS).toBeGreaterThan(d1.conS);
    expect(d1.rom).toBeGreaterThan(90);
    expect(s.toApiReps()[0]).toMatchObject({ set: 1, features: expect.objectContaining({ min_angle: expect.any(Number) }) });
  });

  it("flags forward lean and colours the torso joints", () => {
    const s = new PoseSession("squat");
    const outs = feed(s, repSeries(174, 60, 1.4, 1.0), { extraLean: 35 });
    expect(s.reps).toHaveLength(1);
    expect(s.reps[0].faults).toContain("forward_lean");
    expect(outs.some((o) => o.badJoints.includes(11))).toBe(true);
    expect(outs.some((o) => o.cue === "Chest up")).toBe(true);
  });

  it("gives a 'go lower' cue for partial reps and tracks sets", () => {
    const s = new PoseSession("squat");
    const outs = feed(s, repSeries(174, 135, 1.0, 1.0));
    expect(s.reps).toHaveLength(0);
    expect(outs.some((o) => o.cue === "Go lower")).toBe(true);
    s.nextSet();
    feed(s, repSeries(174, 60, 1.4, 1.0), {}, 4, 10);
    expect(s.reps[0].set).toBe(2);
    expect(s.setReps).toBe(1);
  });

  it("does not count while the body is not visible", () => {
    const s = new PoseSession("squat");
    const outs = feed(s, repSeries(174, 60, 1.4, 1.0), { visibility: 0.2 });
    expect(s.reps).toHaveLength(0);
    expect(outs[0].message).toMatch(/step back/i);
  });
});

describe("CueThrottle", () => {
  it("rate limits and suppresses repeats", () => {
    const c = new CueThrottle(1.8, 4, 0.8);
    expect(c.offer("Chest up", 0)?.text).toBe("Chest up");
    expect(c.offer("Go lower", 1)).toBeNull();          // too soon
    expect(c.offer("Go lower", 1, 3)?.text).toBe("Go lower"); // higher priority interrupts
    expect(c.offer("Chest up", 3.5)).toBeNull();        // same text within 4 s
    expect(c.offer("Chest up", 4.5)?.text).toBe("Chest up");
  });
});
