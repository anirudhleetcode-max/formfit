// Runs the in-app exercise recognizer + rep gate on a landmark file (real-video sanity check).
//   npx tsx scripts/recognize-clip.ts <landmarks.json> <exercise>
import { readFileSync } from "node:fs";
import { PoseSession } from "../src/pose/engine";
import { DEFAULT_GATE_TAU, ExerciseRecognizer, type TcnModel } from "../src/pose/exercise";
import type { ExerciseId } from "../src/pose/rules";

const [file, ex] = process.argv.slice(2) as [string, ExerciseId];
const model: TcnModel = JSON.parse(readFileSync(new URL("../public/models/exercise_tcn.json", import.meta.url), "utf8"));
const data = JSON.parse(readFileSync(file, "utf8"));
const aspect = data.width / data.height;
const s = new PoseSession(ex, { guard: true, recognizer: new ExerciseRecognizer(model), gateTau: DEFAULT_GATE_TAU });
const seen: Record<string, number> = {};
for (const fr of data.frames) {
  const out = s.process(fr.lm ? fr.lm.map(([x, y, v]: number[]) => ({ x, y, visibility: v })) : null, fr.t, aspect);
  if (out.recognized) seen[out.recognized.label] = (seen[out.recognized.label] ?? 0) + 1;
}
console.log(JSON.stringify({ selected: ex, reps: s.reps.length, vetoed: s.vetoedReps, recognized_frames: seen }));
