// Runs recorded MediaPipe landmarks through the SAME TypeScript pipeline the app uses
// (src/pose/engine.ts) and prints one JSON result per clip.
//
//   npx tsx scripts/eval-clips.ts <manifest.json> <landmarks-dir> [out.json]
//
// Landmark files are produced by `python -m experiments.real_clips.extract` and look like
// { width, height, fps, frames: [{ t, lm: [[x, y, visibility] x33] | null }] }.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Point } from "../src/pose/angles";
import { PoseSession } from "../src/pose/engine";
import { toApiFeatures, type ExerciseId } from "../src/pose/rules";

type Clip = { id: string; exercise: ExerciseId; gt_reps: number | null };
type Frames = { width: number; height: number; fps: number; frames: { t: number; lm: number[][] | null }[] };

const [manifestPath, lmDir, outPath] = process.argv.slice(2);
if (!manifestPath || !lmDir) {
  console.error("usage: tsx scripts/eval-clips.ts <manifest.json> <landmarks-dir> [out.json]");
  process.exit(2);
}
const manifest: { clips: Clip[] } = JSON.parse(readFileSync(manifestPath, "utf8"));
const results = [];
for (const clip of manifest.clips) {
  const f = join(lmDir, `${clip.id}.json`);
  if (!existsSync(f)) {
    results.push({ id: clip.id, exercise: clip.exercise, error: "landmarks missing" });
    continue;
  }
  const data: Frames = JSON.parse(readFileSync(f, "utf8"));
  const s = new PoseSession(clip.exercise);
  const aspect = data.width / data.height;
  let visible = 0, confSum = 0;
  for (const fr of data.frames) {
    const lms: Point[] | null = fr.lm ? fr.lm.map(([x, y, v]) => ({ x, y, visibility: v })) : null;
    const out = s.process(lms, fr.t, aspect);
    if (out.visible) visible += 1;
    confSum += out.confidence;
  }
  const scored = s.reps.filter((r) => r.scored);
  results.push({
    id: clip.id,
    exercise: clip.exercise,
    frames: data.frames.length,
    frames_with_pose: data.frames.filter((x) => x.lm).length,
    frames_usable: visible,
    mean_frame_confidence: Math.round((confSum / Math.max(1, data.frames.length)) * 100) / 100,
    gt_reps: clip.gt_reps,
    pred_reps: s.reps.length,
    scored_reps: scored.length,
    reps: s.reps.map((r) => ({
      n: r.n, t: r.t, score: r.score, faults: r.faults, confidence: r.confidence,
      abstain: r.abstainReason, min_angle: r.features.minAngle, max_angle: r.features.maxAngle,
      ecc_s: r.eccS, con_s: r.conS, scored: r.scored, api_features: toApiFeatures(r.features),
    })),
  });
}
const json = JSON.stringify(results, null, 1);
if (outPath) writeFileSync(outPath, json);
else console.log(json);
