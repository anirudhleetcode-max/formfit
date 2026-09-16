// Downloads the MediaPipe pose model (pose_landmarker_lite.task, ~5.8 MB) into public/models/.
// Skips the download when the file is already there. Usage: `npm run fetch-model`
import { createWriteStream, existsSync, mkdirSync, renameSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";

const URL_ = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "public", "models", "pose_landmarker_lite.task");

if (existsSync(out) && statSync(out).size > 1_000_000) {
  console.log("[fetch-model] model present:", out);
  process.exit(0);
}
mkdirSync(dirname(out), { recursive: true });
console.log("[fetch-model] downloading", URL_);
const res = await fetch(URL_);
if (!res.ok || !res.body) {
  console.error(`[fetch-model] download failed (${res.status}). Download the file manually from\n  ${URL_}\nand save it as ${out}`);
  process.exit(1);
}
await pipeline(Readable.fromWeb(res.body), createWriteStream(out + ".part"));
renameSync(out + ".part", out);
console.log(`[fetch-model] saved ${(statSync(out).size / 1e6).toFixed(1)} MB -> ${out}`);
