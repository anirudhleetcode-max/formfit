// Exercise recognition from the last 2 s of landmarks with a small temporal convolutional network.
//
// A TypeScript port of backend/cv/normalize.py (features) and backend/cv/classify.py (TCN forward pass).
// The weights are trained offline on synthetic skeleton sequences (python -m experiments.run
// --config experiments/configs/cv_classification.json) and exported to public/models/exercise_tcn.json
// by `python -m cv.export_tcn`. A vitest parity test checks this file against NumPy.
//
// Used for two things: telling the user when the movement does not look like the selected exercise,
// and vetoing reps that the counter would otherwise take from a different movement (false reps).

import type { Point } from "./angles";

export type TcnModel = {
  classes: string[];
  window: number;
  layers: { kernel: number; dilation: number }[];
  mu: number[];
  sd: number[];
  params: Record<string, number[] | number[][] | number[][][]>;
};

const L_SH = 11, R_SH = 12, L_EL = 13, R_EL = 14, L_WR = 15, R_WR = 16;
const L_HIP = 23, R_HIP = 24, L_KNEE = 25, R_KNEE = 26, L_ANK = 27, R_ANK = 28;
const BODY12 = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK];

type V = [number, number];
const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1]];
const midp = (a: V, b: V): V => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
const norm = (a: V) => Math.hypot(a[0], a[1]);

function angle(a: V, b: V, c: V): number {
  const v1 = sub(a, b), v2 = sub(c, b);
  const cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (norm(v1) * norm(v2) + 1e-9);
  return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
}

function fromVertical(lo: V, hi: V): number {
  const d = sub(hi, lo);
  return (Math.atan2(d[0], -d[1]) * 180) / Math.PI;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

/** Raw frames ([x, y, visibility] x 33, or null) -> [T][51] features, identical to normalize.frame_features. */
export function windowFeatures(frames: (number[][] | null)[], aspect: number): number[][] {
  const T = frames.length;
  // forward fill missing frames (visibility 0), then back-fill the leading gap from the first frame
  const filled: (number[][] | null)[] = new Array(T).fill(null);
  let last: number[][] | null = null;
  for (let i = 0; i < T; i++) {
    const f = frames[i];
    if (f) { last = f; filled[i] = f; }
    else if (last) filled[i] = last.map(([x, y]) => [x, y, 0]);
  }
  const first = frames.findIndex((f) => f !== null);
  if (first > 0) for (let i = 0; i < first; i++) filled[i] = frames[first]!.map(([x, y]) => [x, y, 0]);
  const lm = filled.map((f) => f ?? Array.from({ length: 33 }, () => [0, 0, 0]));

  const xy: V[][] = lm.map((f) => {
    const pts = f.map(([x, y]) => [x * aspect, y] as V);
    const hip = midp(pts[L_HIP], pts[R_HIP]);
    return pts.map((p) => sub(p, hip));
  });
  const sizes = xy.map((P) => {
    const sh = midp(P[L_SH], P[R_SH]), hip = midp(P[L_HIP], P[R_HIP]), ank = midp(P[L_ANK], P[R_ANK]);
    return Math.max(norm(sub(sh, hip)), 0.5 * norm(sub(hip, ank)), norm(sub(P[L_SH], P[R_SH])));
  });
  const s = Math.max(median(sizes) || 1, 1e-6);

  return xy.map((raw, t) => {
    const P = raw.map(([x, y]) => [x / s, y / s] as V);
    const vis = lm[t].map((p) => p[2]);
    const sh = midp(P[L_SH], P[R_SH]), hip = midp(P[L_HIP], P[R_HIP]);
    const trunk = (fromVertical(hip, sh) * Math.PI) / 180;
    const torso = norm(sub(sh, hip)) + 1e-6;
    const out = [
      ...BODY12.map((i) => P[i][0]), ...BODY12.map((i) => P[i][1]), ...BODY12.map((i) => vis[i]),
      angle(P[L_HIP], P[L_KNEE], P[L_ANK]) / 180, angle(P[R_HIP], P[R_KNEE], P[R_ANK]) / 180,
      angle(P[L_SH], P[L_HIP], P[L_KNEE]) / 180, angle(P[R_SH], P[R_HIP], P[R_KNEE]) / 180,
      angle(P[L_SH], P[L_EL], P[L_WR]) / 180, angle(P[R_SH], P[R_EL], P[R_WR]) / 180,
      angle(P[L_EL], P[L_SH], P[L_HIP]) / 180, angle(P[R_EL], P[R_SH], P[R_HIP]) / 180,
      Math.sin(trunk), Math.cos(trunk),
      Math.abs(fromVertical(P[L_ANK], P[L_KNEE])) / 180, Math.abs(fromVertical(P[R_ANK], P[R_KNEE])) / 180,
      norm(sub(P[L_SH], P[R_SH])) / torso / 2,
      midp(P[L_WR], P[R_WR])[1] - sh[1],
      frames[t] ? 1 : 0,
    ];
    return out.map((v) => (Number.isFinite(v) ? v : 0));
  });
}

/** TCN forward pass: features [T][F] -> class probabilities. */
export function tcnProbs(model: TcnModel, feats: number[][]): number[] {
  const T = feats.length;
  let h = feats.map((row) => row.map((v, j) => (v - model.mu[j]) / model.sd[j]));
  model.layers.forEach(({ kernel: k, dilation: d }, li) => {
    const w = model.params[`w${li}`] as number[][][];
    const b = model.params[`b${li}`] as number[];
    const cout = b.length, cin = h[0].length, pad = (d * (k - 1)) / 2;
    const out: number[][] = [];
    for (let t = 0; t < T; t++) {
      const o = b.slice();
      for (let j = 0; j < k; j++) {
        const src = t + j * d - pad;
        if (src < 0 || src >= T) continue;
        const x = h[src], wj = w[j];
        for (let c = 0; c < cin; c++) {
          const xc = x[c];
          if (xc === 0) continue;
          const wr = wj[c];
          for (let q = 0; q < cout; q++) o[q] += xc * wr[q];
        }
      }
      out.push(o.map((v) => Math.max(0, v)));
    }
    h = out;
  });
  const C = h[0].length;
  const pooled: number[] = [];
  for (let c = 0; c < C; c++) pooled.push(h.reduce((a, r) => a + r[c], 0) / T);
  for (let c = 0; c < C; c++) pooled.push(Math.max(...h.map((r) => r[c])));
  const wo = model.params.wo as number[][], bo = model.params.bo as number[];
  const logits = bo.map((bv, q) => pooled.reduce((a, z, i) => a + z * wo[i][q], bv));
  const m = Math.max(...logits);
  const e = logits.map((z) => Math.exp(z - m));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

/** Probability another movement must reach before a counted rep is vetoed (chosen on synthetic data,
 *  experiments/results/*-cv_repcount: false reps vs true reps lost). */
export const DEFAULT_GATE_TAU = 0.8;

export type Recognition = { label: string; p: number; probs: Record<string, number> };

/** Keeps the last `window` analysed frames and classifies them. Call push() once per analysed frame. */
export class ExerciseRecognizer {
  private buf: (number[][] | null)[] = [];
  private aspect = 1;
  private next = -Infinity;
  readonly model: TcnModel;
  constructor(model: TcnModel) { this.model = model; }

  /** Live video runs at 30-60 fps but the model was trained on 15 fps windows: frames that arrive
   *  sooner than 1/15 s after the last kept frame are skipped when `t` (seconds) is given. */
  push(lms: Point[] | null | undefined, aspect: number, t?: number) {
    if (t !== undefined) {
      if (t < this.next - 0.01) return;
      this.next = t + 1 / 15;
    }
    this.aspect = aspect;
    this.buf.push(lms && lms.length >= 33 ? lms.map((p) => [p.x, p.y, p.visibility ?? 0]) : null);
    if (this.buf.length > this.model.window) this.buf.shift();
  }

  ready(): boolean {
    return this.buf.length === this.model.window && this.buf.some((f) => f !== null);
  }

  recognize(): Recognition | null {
    if (!this.ready()) return null;
    const p = tcnProbs(this.model, windowFeatures(this.buf, this.aspect));
    let best = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
    return { label: this.model.classes[best], p: p[best], probs: Object.fromEntries(this.model.classes.map((c, i) => [c, p[i]])) };
  }
}

let loading: Promise<TcnModel | null> | null = null;
/** Fetches the exported weights once. Resolves to null if they cannot be loaded (recognition is optional). */
export function loadExerciseModel(): Promise<TcnModel | null> {
  loading ??= fetch(`${import.meta.env.BASE_URL}models/exercise_tcn.json`)
    .then((r) => (r.ok ? (r.json() as Promise<TcnModel>) : null))
    .catch(() => null);
  return loading;
}
