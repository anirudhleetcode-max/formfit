// Batch runner for the computer-vision experiments (experiments/cv): feeds landmark files through
// the app's own PoseSession (src/pose/engine.ts) and writes one result per job.
//
//   npx tsx scripts/eval-synthetic.ts <jobs.json> <out.json>
//
// jobs.json: { "jobs": [{ "id", "file", "exercise", "guard": false | true | {GuardOptions} }] }
// Landmark files use the eval-clips format: { width, height, fps, frames: [{ t, lm: [[x, y, vis] x33] | null }] }.
import { readFileSync, writeFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import type { Point } from "../src/pose/angles";
import { PoseSession } from "../src/pose/engine";
import type { ExerciseId } from "../src/pose/rules";

type Job = { id: string; file: string; exercise: ExerciseId; guard?: boolean | Record<string, number | boolean> };
type Frames = { width: number; height: number; frames: { t: number; lm: number[][] | null }[] };

const [jobsPath, outPath] = process.argv.slice(2);
if (!jobsPath || !outPath) {
  console.error("usage: tsx scripts/eval-synthetic.ts <jobs.json> <out.json>");
  process.exit(2);
}
const { jobs }: { jobs: Job[] } = JSON.parse(readFileSync(jobsPath, "utf8"));
const cache = new Map<string, Frames>();
const results = [];
let totalMs = 0, totalFrames = 0;
for (const job of jobs) {
  let data = cache.get(job.file);
  if (!data) {
    data = JSON.parse(readFileSync(job.file, "utf8")) as Frames;
    cache.clear();
    cache.set(job.file, data);
  }
  const guard = job.guard === true ? {} : job.guard || false;
  const s = new PoseSession(job.exercise, { guard });
  const aspect = data.width / data.height;
  let usable = 0;
  const t0 = performance.now();
  for (const fr of data.frames) {
    const lms: Point[] | null = fr.lm ? fr.lm.map(([x, y, v]) => ({ x, y, visibility: v })) : null;
    if (s.process(lms, fr.t, aspect).visible) usable += 1;
  }
  const ms = performance.now() - t0;
  totalMs += ms;
  totalFrames += data.frames.length;
  results.push({
    id: job.id, exercise: job.exercise, frames: data.frames.length, frames_usable: usable,
    ms_per_frame: Math.round((ms / Math.max(1, data.frames.length)) * 1000) / 1000,
    guard: s.guard ? s.guard.stats : null,
    reps: s.reps.map((r) => ({
      t: r.t, ecc_s: r.eccS, con_s: r.conS, score: r.score, faults: r.faults, scored: r.scored,
      confidence: r.confidence, min_angle: r.features.minAngle, max_angle: r.features.maxAngle,
    })),
  });
}
writeFileSync(outPath, JSON.stringify({ ms_per_frame: totalMs / Math.max(1, totalFrames), results }));
console.log(`${results.length} jobs, ${totalFrames} frames, ${(totalMs / Math.max(1, totalFrames)).toFixed(3)} ms/frame`);
