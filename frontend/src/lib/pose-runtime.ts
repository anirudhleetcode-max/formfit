// MediaPipe PoseLandmarker loader + skeleton drawing. Video frames stay in the browser.
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import { EDGES, type Point } from "../pose/angles";

export type Delegate = "GPU" | "CPU";
type Loaded = { landmarker: PoseLandmarker; delegate: Delegate };

let loading: Promise<Loaded> | null = null;
let lastTs = 0;

const WASM_BASE = `${import.meta.env.BASE_URL}mediapipe/wasm`;
const MODEL_URL = `${import.meta.env.BASE_URL}models/pose_landmarker_lite.task`;

function preferredDelegate(): Delegate {
  const q = new URLSearchParams(window.location.search).get("delegate");
  if (q?.toUpperCase() === "CPU") return "CPU";
  return "GPU";
}

async function create(delegate: Delegate): Promise<PoseLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  return PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

/** Loads the model once. Tries the GPU delegate first and falls back to CPU. */
export function getLandmarker(): Promise<Loaded> {
  if (!loading) {
    loading = (async () => {
      const want = preferredDelegate();
      if (want === "GPU") {
        try {
          const lm = await create("GPU");
          // a GPU context can be created yet fail on first use (e.g. blocklisted drivers)
          const probe = document.createElement("canvas");
          probe.width = probe.height = 64;
          lm.detectForVideo(probe, nextTimestamp(0));
          return { landmarker: lm, delegate: "GPU" as Delegate };
        } catch (e) {
          console.warn("[pose] GPU delegate failed, using CPU", e);
        }
      }
      return { landmarker: await create("CPU"), delegate: "CPU" as Delegate };
    })();
    loading.catch(() => { loading = null; });
  }
  return loading;
}

/** VIDEO mode needs strictly increasing timestamps across every caller. */
export function nextTimestamp(ms: number): number {
  lastTs = Math.max(lastTs + 1, Math.floor(ms));
  return lastTs;
}

export function detect(lm: PoseLandmarker, source: HTMLVideoElement, ms: number): Point[] | null {
  const res = lm.detectForVideo(source, nextTimestamp(ms));
  return (res.landmarks?.[0] as Point[] | undefined) ?? null;
}

const COLOR_OK = "rgba(236, 240, 243, 0.9)";
const COLOR_JOINT = "#ff5a1f";
const COLOR_BAD = "#ff3040";

export function drawSkeleton(ctx: CanvasRenderingContext2D, lms: Point[] | null, bad: Set<number>) {
  const { width: w, height: h } = ctx.canvas;
  ctx.clearRect(0, 0, w, h);
  if (!lms) return;
  const scale = Math.max(1.5, w / 480);
  const visible = (i: number) => (lms[i]?.visibility ?? 0) > 0.35;
  ctx.lineCap = "round";
  for (const [a, b] of EDGES) {
    if (!visible(a) || !visible(b)) continue;
    const isBad = bad.has(a) || bad.has(b);
    ctx.strokeStyle = isBad ? COLOR_BAD : COLOR_OK;
    ctx.lineWidth = (isBad ? 3.2 : 2.2) * scale;
    ctx.beginPath();
    ctx.moveTo(lms[a].x * w, lms[a].y * h);
    ctx.lineTo(lms[b].x * w, lms[b].y * h);
    ctx.stroke();
  }
  for (let i = 11; i < 33; i++) {
    if (!visible(i)) continue;
    const isBad = bad.has(i);
    ctx.fillStyle = isBad ? COLOR_BAD : COLOR_JOINT;
    ctx.beginPath();
    ctx.arc(lms[i].x * w, lms[i].y * h, (isBad ? 6 : 4) * scale, 0, Math.PI * 2);
    ctx.fill();
  }
}
