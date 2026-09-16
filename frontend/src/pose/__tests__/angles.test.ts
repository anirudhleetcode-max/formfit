import { describe, expect, it } from "vitest";
import { Ema, OneEuro, angleDeg, angleFromVertical, bestSide, lineOffset } from "../angles";
import { gauss, rng } from "./synthetic";

describe("angles", () => {
  it("computes interior angles", () => {
    expect(angleDeg({ x: 0, y: 1 }, { x: 0, y: 0 }, { x: 1, y: 0 })).toBeCloseTo(90);
    expect(angleDeg({ x: -1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 })).toBeCloseTo(180);
    expect(angleDeg({ x: 1, y: 1 }, { x: 0, y: 0 }, { x: 1, y: 0 })).toBeCloseTo(45);
    expect(Number.isNaN(angleDeg({ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 }))).toBe(true);
  });

  it("measures lean from vertical in image coordinates", () => {
    expect(angleFromVertical({ x: 0.5, y: 0.8 }, { x: 0.5, y: 0.4 })).toBeCloseTo(0);
    expect(angleFromVertical({ x: 0.5, y: 0.8 }, { x: 0.9, y: 0.4 })).toBeCloseTo(45);
    expect(angleFromVertical({ x: 0.5, y: 0.8 }, { x: 0.1, y: 0.4 })).toBeCloseTo(45);
  });

  it("signs hip offset the same way for both facing directions", () => {
    const sag = { x: 0.5, y: 0.62 }, pike = { x: 0.5, y: 0.5 };
    const a = { x: 0.2, y: 0.55 }, c = { x: 0.8, y: 0.6 };
    expect(lineOffset(a, c, sag)).toBeGreaterThan(0);
    expect(lineOffset(c, a, sag)).toBeGreaterThan(0);
    expect(lineOffset(a, c, pike)).toBeLessThan(0);
    expect(lineOffset(c, a, pike)).toBeLessThan(0);
  });

  it("picks the more visible side", () => {
    const lms = Array.from({ length: 33 }, (_, i) => ({ x: 0, y: 0, visibility: i % 2 ? 0.9 : 0.2 }));
    expect(bestSide(lms, [11, 23], [12, 24])).toBe("left"); // 11,23 are odd -> 0.9
    expect(bestSide(lms, [12, 24], [11, 23])).toBe("right");
  });
});

describe("filters", () => {
  it("EMA converges and ignores NaN", () => {
    const f = new Ema(0.5);
    let v = 0;
    for (let i = 0; i < 20; i++) v = f.next(10);
    expect(v).toBeCloseTo(10);
    expect(f.next(NaN)).toBeCloseTo(10);
  });

  it("One Euro reduces jitter on a still signal but tracks a fast move", () => {
    const r = rng(1);
    const f = new OneEuro(1.2, 0.015);
    const raw: number[] = [], out: number[] = [];
    for (let i = 0; i < 90; i++) {
      const x = 170 + 3 * gauss(r);
      raw.push(x);
      out.push(f.next(x, i / 30));
    }
    const sd = (a: number[]) => { const m = a.reduce((s, x) => s + x, 0) / a.length; return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length); };
    expect(sd(out.slice(30))).toBeLessThan(sd(raw.slice(30)) * 0.6);
    // step to 80 degrees: within 0.3 s the filter should be most of the way there
    let y = 0;
    for (let i = 0; i < 9; i++) y = f.next(80, 3 + i / 30);
    expect(y).toBeLessThan(110);
  });
});
