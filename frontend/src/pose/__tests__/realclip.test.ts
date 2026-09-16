// Regression test on REAL landmarks: MediaPipe output recorded from samples/squat_demo.webm
// (4 barbell back squats filmed from behind at an angle) by backend/scripts/dump_landmarks.py.
import { readFileSync, writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { Point } from "../angles";
import { PoseSession } from "../engine";

type Fixture = { width: number; height: number; frames: { t: number; lm: number[][] | null }[] };
const fx: Fixture = JSON.parse(readFileSync(new URL("./fixtures/squat_demo.json", import.meta.url), "utf8"));

function run() {
  const s = new PoseSession("squat");
  const aspect = fx.width / fx.height;
  for (const f of fx.frames) {
    const lms: Point[] | null = f.lm ? f.lm.map(([x, y, v]) => ({ x, y, visibility: v })) : null;
    s.process(lms, f.t, aspect);
  }
  return s;
}

describe("real clip: squat_demo.webm", () => {
  it("finds a pose in almost every frame", () => {
    const withPose = fx.frames.filter((f) => f.lm).length;
    expect(withPose / fx.frames.length).toBeGreaterThan(0.9);
  });

  it("counts the four squats", () => {
    const s = run();
    if (process.env.DEBUG_CLIP) writeFileSync(process.env.DEBUG_CLIP, JSON.stringify(s.reps, null, 1));
    expect(s.reps).toHaveLength(4);
    for (const r of s.reps) {
      expect(r.features.minAngle).toBeLessThan(80);      // deep squats
      expect(r.eccS).toBeGreaterThan(0.5);
      expect(r.conS).toBeGreaterThan(0.4);
    }
  });

  it("scores the demonstrator's reps as clean (no depth, heel or lockout false alarms)", () => {
    const s = run();
    for (const r of s.reps) {
      expect(r.scored).toBe(true);
      expect(r.confidence).toBeGreaterThan(0.6);
      expect(r.faults).toEqual([]);
      expect(r.features.maxAngle).toBeGreaterThan(170);
      expect(r.faults).not.toContain("shallow_depth");
      expect(r.faults).not.toContain("heel_lift");
      expect(r.features.valgusMin).toBeGreaterThan(0.8);
    }
  });
});
