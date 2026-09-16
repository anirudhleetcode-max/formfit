// Per-exercise definitions: which joint angle drives the rep counter, per-frame metrics,
// live (per-frame) form checks and the per-rep form score.
// Rep thresholds are mirrored in backend/ml/rules.py (a backend test checks they stay in sync).

import {
  LM, angleDeg, angleFromVertical, bestSide, dist, lineOffset, mid, minVisibility, type Point,
} from "./angles";
import type { FsmConfig } from "./fsm";

export type ExerciseId = "squat" | "pushup" | "curl" | "press" | "lunge";

export type FrameMetrics = {
  primary: number;          // driving joint angle (deg)
  torsoLean: number;        // hip->shoulder from vertical (deg)
  hipLine: number;          // shoulder-hip-ankle angle (deg)
  hipOffset: number;        // + hips below shoulder-ankle line (sag), - above (pike)
  valgus: number | null;    // knee width / ankle width (frontal view only)
  elbowDrift: number;       // upper arm vs torso (deg)
  depth: number;            // (hip.y - knee.y) / thigh length; > 0 = hip below knee
  heelRise: number;         // heel lift / foot length
  frontal: boolean;
};

export type RepFeatures = {
  minAngle: number; maxAngle: number; rom: number; eccS: number; conS: number;
  torsoLeanMax: number; torsoSway: number; hipLineMin: number; valgusMin: number;
  elbowDriftMax: number; depth: number; heelRise: number;
  hipOffsetAtWorst: number;
};

export type FaultCode =
  | "shallow_depth" | "forward_lean" | "knee_valgus" | "heel_lift" | "rushed_descent" | "no_lockout"
  | "hip_sag" | "hip_pike" | "shallow_pushup" | "elbow_drift" | "torso_swing" | "partial_extension"
  | "partial_curl" | "back_arch" | "short_press" | "torso_lean_lunge" | "shallow_lunge";

// Points off 100 for each fault (mirrored in backend/ml/features.py).
export const PENALTY: Record<FaultCode, number> = {
  shallow_depth: 30,
  forward_lean: 20,
  knee_valgus: 25,
  heel_lift: 15,
  rushed_descent: 10,
  no_lockout: 15,
  hip_sag: 30,
  hip_pike: 20,
  shallow_pushup: 30,
  elbow_drift: 25,
  torso_swing: 25,
  partial_extension: 20,
  partial_curl: 20,
  back_arch: 30,
  short_press: 20,
  torso_lean_lunge: 20,
  shallow_lunge: 30,
};

export const FAULT_INFO: Record<FaultCode, { label: string; cue: string }> = {
  shallow_depth: { label: "Not deep enough", cue: "Go lower" },
  forward_lean: { label: "Chest dropping", cue: "Chest up" },
  knee_valgus: { label: "Knees caving", cue: "Push your knees out" },
  heel_lift: { label: "Heels lifting", cue: "Keep your heels down" },
  rushed_descent: { label: "Rushed descent", cue: "Slow down on the way down" },
  no_lockout: { label: "No lockout", cue: "Finish tall at the top" },
  hip_sag: { label: "Hips sagging", cue: "Squeeze your glutes, lift the hips" },
  hip_pike: { label: "Hips piking", cue: "Lower your hips" },
  shallow_pushup: { label: "Shallow push-up", cue: "Chest to the floor" },
  elbow_drift: { label: "Elbows drifting", cue: "Pin your elbows" },
  torso_swing: { label: "Swinging torso", cue: "Stop swinging" },
  partial_extension: { label: "Partial extension", cue: "Straighten the arm fully" },
  partial_curl: { label: "Partial curl", cue: "Squeeze all the way up" },
  back_arch: { label: "Lower back arching", cue: "Ribs down, brace" },
  short_press: { label: "Press not locked out", cue: "Lock it out overhead" },
  torso_lean_lunge: { label: "Leaning forward", cue: "Stay tall" },
  shallow_lunge: { label: "Lunge too shallow", cue: "Drop the back knee" },
};

export type LiveFault = { code: FaultCode; joints: number[] };

export type ExerciseDef = {
  id: ExerciseId;
  label: string;
  /** joints that must be visible (both sides listed; the best side is used when `oneSide`) */
  left: number[];
  right: number[];
  oneSide: boolean;
  /** transforms the primary angle into the FSM signal (rest position = high) */
  signal: (primary: number) => number;
  fsm: Partial<FsmConfig>;
  /** the first half of the FSM rep (signal going down) is the concentric phase */
  concentricFirst: boolean;
  setup: string;
  liveChecks: (m: FrameMetrics) => LiveFault[];
  evaluate: (f: RepFeatures) => FaultCode[];
};

const LEG_L = [LM.lShoulder, LM.lHip, LM.lKnee, LM.lAnkle];
const LEG_R = [LM.rShoulder, LM.rHip, LM.rKnee, LM.rAnkle];
const ARM_L = [LM.lShoulder, LM.lElbow, LM.lWrist, LM.lHip];
const ARM_R = [LM.rShoulder, LM.rElbow, LM.rWrist, LM.rHip];
const PLANK_L = [LM.lShoulder, LM.lElbow, LM.lWrist, LM.lHip, LM.lAnkle];
const PLANK_R = [LM.rShoulder, LM.rElbow, LM.rWrist, LM.rHip, LM.rAnkle];
const TORSO = [LM.lShoulder, LM.rShoulder, LM.lHip, LM.rHip];
const KNEES = [LM.lKnee, LM.rKnee];
const HEELS = [LM.lHeel, LM.rHeel];
const ELBOWS = [LM.lElbow, LM.rElbow];

export const EXERCISES: Record<ExerciseId, ExerciseDef> = {
  squat: {
    id: "squat", label: "Squat", left: LEG_L, right: LEG_R, oneSide: true,
    signal: (a) => a,
    fsm: { top: 150, bottom: 115, hysteresis: 8, minRepS: 0.7, partialMinDrop: 18 },
    concentricFirst: false,
    setup: "Side-on or 45°, whole body in frame, feet shoulder-width.",
    liveChecks: (m) => {
      const out: LiveFault[] = [];
      if (m.primary < 140 && m.torsoLean > 45) out.push({ code: "forward_lean", joints: TORSO });
      if (m.primary < 140 && m.valgus !== null && m.valgus < 0.8) out.push({ code: "knee_valgus", joints: KNEES });
      if (m.heelRise > 0.2) out.push({ code: "heel_lift", joints: HEELS });
      return out;
    },
    evaluate: (f) => {
      const faults: FaultCode[] = [];
      if (f.depth < -0.15) faults.push("shallow_depth");
      if (f.torsoLeanMax > 45) faults.push("forward_lean");
      if (f.valgusMin < 0.8) faults.push("knee_valgus");
      if (f.heelRise > 0.2) faults.push("heel_lift");
      if (f.eccS < 0.5) faults.push("rushed_descent");
      if (f.maxAngle < 160) faults.push("no_lockout");
      return faults;
    },
  },
  pushup: {
    id: "pushup", label: "Push-up", left: PLANK_L, right: PLANK_R, oneSide: true,
    signal: (a) => a,
    fsm: { top: 145, bottom: 110, hysteresis: 8, minRepS: 0.6, partialMinDrop: 15 },
    concentricFirst: false,
    setup: "Side-on to the camera, head to heels in frame.",
    liveChecks: (m) => {
      const out: LiveFault[] = [];
      if (m.hipLine < 160 && m.hipOffset > 0) out.push({ code: "hip_sag", joints: [LM.lHip, LM.rHip] });
      else if (m.hipLine < 160) out.push({ code: "hip_pike", joints: [LM.lHip, LM.rHip] });
      return out;
    },
    evaluate: (f) => {
      const faults: FaultCode[] = [];
      if (f.hipLineMin < 160) faults.push(f.hipOffsetAtWorst >= 0 ? "hip_sag" : "hip_pike");
      if (f.minAngle > 95) faults.push("shallow_pushup");
      if (f.maxAngle < 155) faults.push("no_lockout");
      return faults;
    },
  },
  curl: {
    id: "curl", label: "Bicep curl", left: ARM_L, right: ARM_R, oneSide: true,
    signal: (a) => a,
    fsm: { top: 140, bottom: 80, hysteresis: 10, minRepS: 0.6, partialMinDrop: 25 },
    concentricFirst: true,
    setup: "Side-on, standing tall, working arm closest to the camera.",
    liveChecks: (m) => {
      const out: LiveFault[] = [];
      if (m.elbowDrift > 28) out.push({ code: "elbow_drift", joints: ELBOWS });
      if (m.torsoLean > 12) out.push({ code: "torso_swing", joints: TORSO });
      return out;
    },
    evaluate: (f) => {
      const faults: FaultCode[] = [];
      if (f.elbowDriftMax > 28) faults.push("elbow_drift");
      if (f.torsoSway > 11) faults.push("torso_swing");
      if (f.maxAngle < 150) faults.push("partial_extension");
      if (f.minAngle > 70) faults.push("partial_curl");
      return faults;
    },
  },
  press: {
    id: "press", label: "Shoulder press", left: ARM_L, right: ARM_R, oneSide: false,
    // rest position is the rack (small elbow angle) -> invert so rest is high
    signal: (a) => 180 - a,
    fsm: { top: 80, bottom: 35, hysteresis: 8, minRepS: 0.6, partialMinDrop: 20 },
    concentricFirst: true,
    setup: "Facing the camera or 45°, hips to hands in frame.",
    liveChecks: (m) => (m.torsoLean > 13 ? [{ code: "back_arch", joints: TORSO }] : []),
    evaluate: (f) => {
      const faults: FaultCode[] = [];
      if (f.maxAngle < 158) faults.push("short_press");
      if (f.torsoLeanMax > 13) faults.push("back_arch");
      return faults;
    },
  },
  lunge: {
    id: "lunge", label: "Lunge", left: LEG_L, right: LEG_R, oneSide: false,
    signal: (a) => a,
    fsm: { top: 150, bottom: 120, hysteresis: 8, minRepS: 0.8, partialMinDrop: 15 },
    concentricFirst: false,
    setup: "Side-on, both legs visible through the whole step.",
    liveChecks: (m) => {
      const out: LiveFault[] = [];
      if (m.torsoLean > 22) out.push({ code: "torso_lean_lunge", joints: TORSO });
      if (m.primary < 140 && m.valgus !== null && m.valgus < 0.8) out.push({ code: "knee_valgus", joints: KNEES });
      return out;
    },
    evaluate: (f) => {
      const faults: FaultCode[] = [];
      if (f.minAngle > 108) faults.push("shallow_lunge");
      if (f.torsoLeanMax > 22) faults.push("torso_lean_lunge");
      if (f.valgusMin < 0.8) faults.push("knee_valgus");
      return faults;
    },
  },
};

export const EXERCISE_LIST: ExerciseDef[] = Object.values(EXERCISES);

// Neutral values for features that cannot be measured (mirrors backend/ml/features.py NEUTRAL)
export const NEUTRAL = { valgusMin: 1.0, depth: 0.0, heelRise: 0.0 };

export function scoreFaults(faults: FaultCode[]): number {
  return Math.max(0, 100 - faults.reduce((s, f) => s + PENALTY[f], 0));
}

export function evaluateRep(ex: ExerciseId, f: RepFeatures): { score: number; faults: FaultCode[] } {
  const faults = EXERCISES[ex].evaluate(f);
  return { score: scoreFaults(faults), faults };
}

/** Joints the exercise needs on the chosen side (or both sides). */
export function requiredJoints(ex: ExerciseId, side: "left" | "right"): number[] {
  const d = EXERCISES[ex];
  return d.oneSide ? (side === "left" ? d.left : d.right) : [...d.left, ...d.right];
}

const inFrame = (p: Point | undefined) => !!p && p.x > -0.02 && p.x < 1.02 && p.y > -0.02 && p.y < 1.02;

export type VisibilityResult = { ok: true; side: "left" | "right" } | { ok: false; reason: string };

/** Are the joints this exercise needs visible and inside the frame? */
export function checkVisibility(ex: ExerciseId, lms: Point[] | null | undefined, minVis = 0.5): VisibilityResult {
  if (!lms || lms.length < 33) return { ok: false, reason: "No one in view" };
  const d = EXERCISES[ex];
  const side = bestSide(lms, d.left, d.right);
  const need = requiredJoints(ex, side);
  const low = minVisibility(lms, need) < minVis || need.some((i) => !inFrame(lms[i]));
  if (low) return { ok: false, reason: "Step back so your whole body is visible" };
  return { ok: true, side };
}

/** Raw (unsmoothed) per-frame metrics from 33 landmarks in aspect-corrected image units
 *  (x multiplied by width/height, see PoseSession.process). */
export function frameMetrics(ex: ExerciseId, lms: Point[], side: "left" | "right"): FrameMetrics {
  const L = side === "left";
  const sh = lms[L ? LM.lShoulder : LM.rShoulder];
  const el = lms[L ? LM.lElbow : LM.rElbow];
  const wr = lms[L ? LM.lWrist : LM.rWrist];
  const hip = lms[L ? LM.lHip : LM.rHip];
  const kn = lms[L ? LM.lKnee : LM.rKnee];
  const an = lms[L ? LM.lAnkle : LM.rAnkle];
  const heel = lms[L ? LM.lHeel : LM.rHeel];
  const toe = lms[L ? LM.lFoot : LM.rFoot];

  const shoulderW = dist(lms[LM.lShoulder], lms[LM.rShoulder]);
  const torsoLen = dist(mid(lms[LM.lShoulder], lms[LM.rShoulder]), mid(lms[LM.lHip], lms[LM.rHip])) || 1e-6;
  // shoulder width / torso length: ~0.1-0.3 side-on, ~0.5-0.9 facing (or back to) the camera
  const ratio = shoulderW / torsoLen;
  const frontal = ratio > 0.45;
  const sideOn = ratio < 0.35;

  const kneeL = angleDeg(lms[LM.lHip], lms[LM.lKnee], lms[LM.lAnkle]);
  const kneeR = angleDeg(lms[LM.rHip], lms[LM.rKnee], lms[LM.rAnkle]);
  const elbowL = angleDeg(lms[LM.lShoulder], lms[LM.lElbow], lms[LM.lWrist]);
  const elbowR = angleDeg(lms[LM.rShoulder], lms[LM.rElbow], lms[LM.rWrist]);

  let primary: number;
  switch (ex) {
    case "squat": primary = frontal ? (kneeL + kneeR) / 2 : angleDeg(hip, kn, an); break;
    case "lunge": primary = Math.min(kneeL, kneeR); break;
    case "press": primary = (elbowL + elbowR) / 2; break;
    default: primary = angleDeg(sh, el, wr);
  }

  const torso = frontal
    ? angleFromVertical(mid(lms[LM.lHip], lms[LM.rHip]), mid(lms[LM.lShoulder], lms[LM.rShoulder]))
    : angleFromVertical(hip, sh);

  let valgus: number | null = null;
  const ankleW = Math.abs(lms[LM.lAnkle].x - lms[LM.rAnkle].x);
  if (frontal && ankleW > 0.03) valgus = Math.abs(lms[LM.lKnee].x - lms[LM.rKnee].x) / ankleW;

  const footLen = dist(heel, toe);
  // heel lift is only readable side-on with the whole foot inside the frame
  const footOk = sideOn && footLen > 0.02 && heel.y < 1 && toe.y < 1 && (heel.visibility ?? 0) > 0.5 && (toe.visibility ?? 0) > 0.5;
  const heelRise = footOk ? Math.max(0, (toe.y - heel.y) / footLen) : 0;
  const thigh = dist(hip, kn) || 1e-6;

  return {
    primary,
    torsoLean: torso,
    hipLine: angleDeg(sh, hip, an),
    hipOffset: lineOffset(sh, an, hip),
    valgus,
    elbowDrift: angleDeg(el, sh, hip),
    depth: (hip.y - kn.y) / thigh,
    heelRise,
    frontal,
  };
}

export function toApiFeatures(f: RepFeatures): Record<string, number> {
  return {
    min_angle: f.minAngle, max_angle: f.maxAngle, rom: f.rom, ecc_s: f.eccS, con_s: f.conS,
    torso_lean_max: f.torsoLeanMax, torso_sway: f.torsoSway, hip_line_min: f.hipLineMin,
    valgus_min: f.valgusMin, elbow_drift_max: f.elbowDriftMax, depth: f.depth, heel_rise: f.heelRise,
  };
}
