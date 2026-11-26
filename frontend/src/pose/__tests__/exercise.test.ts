import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { ExerciseRecognizer, tcnProbs, windowFeatures, type TcnModel } from "../exercise";

// Written by `python -m cv.export_tcn` (NumPy reference values).
const fix = join(__dirname, "fixtures", "tcn_parity.json");
const modelPath = join(__dirname, "..", "..", "..", "public", "models", "exercise_tcn.json");
const ok = existsSync(fix) && existsSync(modelPath);

type Case = { label: string; aspect: number; frames: (number[][] | null)[]; features: number[][]; probs: number[] };

describe.skipIf(!ok)("exercise TCN parity with NumPy", () => {
  const model: TcnModel = ok ? JSON.parse(readFileSync(modelPath, "utf8")) : null!;
  const cases: Case[] = ok ? JSON.parse(readFileSync(fix, "utf8")).cases : [];

  it("computes the same features", () => {
    for (const c of cases) {
      const f = windowFeatures(c.frames, c.aspect);
      f.forEach((row, t) => row.forEach((v, j) => expect(Math.abs(v - c.features[t][j])).toBeLessThan(2e-3)));
    }
  });

  it("computes the same probabilities and labels", () => {
    for (const c of cases) {
      const p = tcnProbs(model, windowFeatures(c.frames, c.aspect));
      p.forEach((v, i) => expect(Math.abs(v - c.probs[i])).toBeLessThan(2e-3));
    }
  });

  it("recognizer needs a full window", () => {
    const r = new ExerciseRecognizer(model);
    const c = cases[0];
    for (let i = 0; i < c.frames.length - 1; i++) {
      const f = c.frames[i];
      r.push(f ? f.map(([x, y, v]) => ({ x, y, visibility: v })) : null, c.aspect);
    }
    expect(r.recognize()).toBeNull();
    const f = c.frames[c.frames.length - 1];
    r.push(f ? f.map(([x, y, v]) => ({ x, y, visibility: v })) : null, c.aspect);
    expect(r.recognize()?.label).toBeDefined();
  });
});
