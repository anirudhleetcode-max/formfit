// Copies the MediaPipe Tasks WASM runtime from node_modules into public/ so the app never
// loads code from a CDN at runtime. Runs automatically before `npm run dev` / `npm run build`.
import { cpSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const dst = join(root, "public", "mediapipe", "wasm");
if (!existsSync(src)) {
  console.error("[copy-wasm] @mediapipe/tasks-vision is not installed — run `npm install` first");
  process.exit(1);
}
mkdirSync(dst, { recursive: true });
for (const f of readdirSync(src)) cpSync(join(src, f), join(dst, f));
console.log(`[copy-wasm] ${readdirSync(src).length} files -> public/mediapipe/wasm`);
