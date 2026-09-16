import { describe, expect, it } from "vitest";
import { RepFsm, type FsmEvent } from "../fsm";
import { FPS, gauss, repSeries, rng } from "./synthetic";

function run(fsm: RepFsm, series: number[], t0 = 0) {
  const events: FsmEvent[] = [];
  series.forEach((s, i) => {
    const e = fsm.update(t0 + i / FPS, s);
    if (e) events.push(e);
  });
  return events;
}

const cfg = { top: 150, bottom: 115, hysteresis: 8, minRepS: 0.7, maxRepS: 12, partialMinDrop: 18 };

describe("RepFsm", () => {
  it("counts clean reps and measures tempo", () => {
    const fsm = new RepFsm(cfg);
    const one = repSeries(172, 70, 1.5, 0.9);
    const events = run(fsm, [...one, ...one, ...one]);
    const reps = events.filter((e) => e.type === "rep");
    expect(reps).toHaveLength(3);
    expect(fsm.count).toBe(3);
    const t = (reps[0] as Extract<FsmEvent, { type: "rep" }>).timing;
    expect(t.minSignal).toBeCloseTo(70, 0);
    // down phase measured from the last top-threshold frame, so a bit shorter than 1.5 s
    expect(t.downS).toBeGreaterThan(0.8);
    expect(t.downS).toBeLessThan(1.8);
    expect(t.upS).toBeGreaterThan(0.4);
    expect(t.upS).toBeLessThan(1.3);
  });

  it("does not double count noise around a threshold (hysteresis)", () => {
    const fsm = new RepFsm(cfg);
    const r = rng(7);
    const series: number[] = [];
    for (let i = 0; i < 300; i++) series.push(150 + 4 * gauss(r)); // hovering at the top threshold
    for (const s of repSeries(170, 80, 1.2, 1.0)) series.push(s + 2 * gauss(r));
    for (let i = 0; i < 300; i++) series.push(116 + 3 * gauss(r)); // hovering near bottom
    const reps = run(fsm, series).filter((e) => e.type === "rep");
    expect(reps).toHaveLength(1);
  });

  it("reports partial reps that never reach the bottom", () => {
    const fsm = new RepFsm(cfg);
    const events = run(fsm, repSeries(170, 125, 1.0, 1.0));
    expect(events.map((e) => e.type)).toEqual(["started", "partial"]);
    expect(fsm.count).toBe(0);
  });

  it("rejects reps faster than the minimum duration", () => {
    const fsm = new RepFsm(cfg);
    const events = run(fsm, repSeries(170, 90, 0.2, 0.2, 0, 0.3));
    expect(events.some((e) => e.type === "rejected")).toBe(true);
    expect(fsm.count).toBe(0);
  });

  it("abandons a rep that stalls at the bottom", () => {
    const fsm = new RepFsm({ ...cfg, maxRepS: 3 });
    const series = [...repSeries(170, 90, 1, 0, 0, 0.3).slice(0, 40), ...Array(150).fill(90)];
    const events = run(fsm, series);
    expect(events.at(-1)).toEqual({ type: "rejected", reason: "too_slow" });
    expect(fsm.state).toBe("idle");
  });

  it("waits for the top position before counting", () => {
    const fsm = new RepFsm(cfg);
    // starts at the bottom (e.g. user walks in crouched)
    const events = run(fsm, [...Array(20).fill(90), ...repSeries(170, 90, 1, 1)]);
    expect(events.filter((e) => e.type === "rep")).toHaveLength(1);
  });

  it("validates thresholds", () => {
    expect(() => new RepFsm({ top: 100, bottom: 95, hysteresis: 8 })).toThrow();
  });
});
