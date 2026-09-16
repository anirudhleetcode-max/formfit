// Landmarks in, reps out. Shared by live camera mode, upload mode and the unit tests.

import { OneEuro, type Point } from "./angles";
import { FRAME_MIN, JitterMeter, jitterScore, repQuality, visibilityScore, weakestPart } from "./confidence";
import { CueThrottle } from "./cues";
import { RepFsm, type FsmState, type RepTiming } from "./fsm";
import {
  EXERCISES, FAULT_INFO, NEUTRAL, checkVisibility, evaluateRep, frameMetrics, requiredJoints, toApiFeatures,
  type ExerciseId, type FaultCode, type FrameMetrics, type RepFeatures,
} from "./rules";

export type RepResult = {
  n: number;              // rep number within the session (1-based)
  set: number;
  score: number | null;   // null = rep counted but form not scored (low tracking confidence)
  confidence: number;     // rep tracking confidence 0..1
  scored: boolean;
  abstainReason: string | null;
  faults: FaultCode[];
  eccS: number;
  conS: number;
  rom: number;
  t: number;              // seconds since session start (rep end)
  features: RepFeatures;
};

export type FrameOutput = {
  visible: boolean;
  message: string | null;
  badJoints: number[];
  primary: number | null;
  state: FsmState;
  setReps: number;
  totalReps: number;
  rep: RepResult | null;
  cue: string | null;
  confidence: number;        // smoothed frame tracking confidence 0..1 (0 when nobody is visible)
  weakest: string | null;    // least visible required body part
  lowConfidence: boolean;    // form feedback suppressed on this frame
};

type Sample = { t: number; conf: number } & FrameMetrics;

const r2 = (x: number) => Math.round(x * 100) / 100;

/** Aggregate the per-frame samples of one rep into the rep feature vector.
 *  The rep window runs from the last top-threshold frame to the return to the top, which never
 *  contains the full lockout itself, so the joint-angle range uses a window that reaches back to
 *  `angleFrom` (the lockout before the rep, but not before the previous rep ended). */
export function aggregateRep(samples: Sample[], timing: RepTiming, concentricFirst: boolean, angleFrom = timing.start): RepFeatures {
  const w = samples.filter((s) => s.t >= timing.start - 1e-6 && s.t <= timing.end + 1e-6);
  const src = w.length ? w : samples;
  let minA = Infinity, maxA = -Infinity, leanMax = -Infinity, leanMin = Infinity;
  let hipMin = Infinity, hipOff = 0, valMin = Infinity, driftMax = -Infinity, depth = -Infinity, heel = 0;
  for (const s of samples) {
    if (s.t < Math.min(angleFrom, timing.start) - 1e-6 || s.t > timing.end + 1e-6) continue;
    minA = Math.min(minA, s.primary);
    maxA = Math.max(maxA, s.primary);
  }
  for (const s of src) {
    leanMax = Math.max(leanMax, s.torsoLean);
    leanMin = Math.min(leanMin, s.torsoLean);
    if (s.hipLine < hipMin) { hipMin = s.hipLine; hipOff = s.hipOffset; }
    if (s.valgus !== null) valMin = Math.min(valMin, s.valgus);
    driftMax = Math.max(driftMax, s.elbowDrift);
    depth = Math.max(depth, s.depth);
    heel = Math.max(heel, s.heelRise);
  }
  const eccS = concentricFirst ? timing.upS : timing.downS;
  const conS = concentricFirst ? timing.downS : timing.upS;
  return {
    minAngle: r2(minA), maxAngle: r2(maxA), rom: r2(maxA - minA),
    eccS: r2(eccS), conS: r2(conS),
    torsoLeanMax: r2(leanMax), torsoSway: r2(leanMax - leanMin),
    hipLineMin: r2(hipMin), hipOffsetAtWorst: r2(hipOff),
    valgusMin: Number.isFinite(valMin) ? r2(valMin) : NEUTRAL.valgusMin,
    elbowDriftMax: r2(driftMax),
    depth: r2(Math.max(-1.5, Math.min(1.5, depth))),
    heelRise: r2(heel),
  };
}

export class PoseSession {
  readonly exercise: ExerciseId;
  set = 1;
  setReps = 0;
  reps: RepResult[] = [];
  framesSeen = 0;
  framesWithPose = 0;

  private fsm: RepFsm;
  private cues = new CueThrottle();
  private filters: Record<string, OneEuro> = {};
  private buf: Sample[] = [];
  private t0: number | null = null;
  private lastSeen = -Infinity;
  private side: "left" | "right" | null = null;
  private otherSide = 0;
  private lastRepEnd = -Infinity;
  private jitter = new JitterMeter();
  private frameTimes: number[] = [];   // every processed frame (for dropout / fps per rep)
  private confEma = 0;
  private weakest = "body";

  constructor(exercise: ExerciseId) {
    this.exercise = exercise;
    this.fsm = new RepFsm(EXERCISES[exercise].fsm);
  }

  private smooth(key: string, v: number, t: number): number {
    // angles move fast during a rep: a little speed adaptation (beta) keeps lag low
    const f = (this.filters[key] ??= new OneEuro(1.2, 0.015, 1.0));
    return f.next(v, t);
  }

  nextSet() {
    this.set += 1;
    this.setReps = 0;
    this.fsm.state = "idle";
    this.cues.reset();
  }

  elapsed(t: number) {
    return this.t0 === null ? 0 : t - this.t0;
  }

  /** lms: MediaPipe normalised landmarks; aspect = video width / height. x and y are normalised
   *  by different lengths, so x is rescaled by the aspect ratio before measuring any angle. */
  process(lms: Point[] | null | undefined, t: number, aspect = 1): FrameOutput {
    this.framesSeen += 1;
    if (this.t0 === null) this.t0 = t;
    const def = EXERCISES[this.exercise];
    const base: FrameOutput = {
      visible: false, message: null, badJoints: [], primary: null, state: this.fsm.state,
      setReps: this.setReps, totalReps: this.reps.length, rep: null, cue: null,
      confidence: 0, weakest: null, lowConfidence: true,
    };
    if (lms && lms.length >= 33) this.framesWithPose += 1;
    this.frameTimes.push(t);

    const v = checkVisibility(this.exercise, lms);
    if (!v.ok) {
      this.confEma = 0.7 * this.confEma;
      base.confidence = r2(this.confEma);
      if (lms && lms.length >= 33) {
        this.weakest = weakestPart(lms, [...EXERCISES[this.exercise].left, ...EXERCISES[this.exercise].right]);
        base.weakest = this.weakest;
      }
      // a short dropout keeps the rep going; a long one resets the state machine
      if (t - this.lastSeen > 1.5 && this.fsm.state !== "idle") this.fsm.state = "idle";
      base.message = v.reason;
      base.cue = this.cues.offer(v.reason, t, 4)?.text ?? null;
      return base;
    }
    this.lastSeen = t;
    // hysteresis on side choice so the skeleton does not flip every frame
    this.otherSide = v.side !== this.side ? this.otherSide + 1 : 0;
    if (this.side === null || this.otherSide >= 10) { this.side = v.side; this.otherSide = 0; }

    const scaled = aspect === 1 ? lms! : lms!.map((p) => ({ ...p, x: p.x * aspect }));
    const raw = frameMetrics(this.exercise, scaled, this.side);
    const m: FrameMetrics = {
      primary: this.smooth("primary", raw.primary, t),
      torsoLean: this.smooth("torso", raw.torsoLean, t),
      hipLine: this.smooth("hipLine", raw.hipLine, t),
      hipOffset: this.smooth("hipOff", raw.hipOffset, t),
      valgus: raw.valgus === null ? null : this.smooth("valgus", raw.valgus, t),
      elbowDrift: this.smooth("drift", raw.elbowDrift, t),
      depth: this.smooth("depth", raw.depth, t),
      heelRise: this.smooth("heel", raw.heelRise, t),
      frontal: raw.frontal,
    };
    this.buf.push({ t, conf: 0, ...m });
    const keepFrom = t - (this.fsm.config.maxRepS + 2);
    while (this.buf.length && this.buf[0].t < keepFrom) this.buf.shift();

    // ---- tracking confidence for this frame ----
    const need = requiredJoints(this.exercise, this.side);
    const noise = this.jitter.update(raw.primary, t);
    const conf = visibilityScore(need.map((i) => lms![i].visibility ?? 0)) * (noise === null ? 1 : jitterScore(noise));
    this.confEma = 0.3 * conf + 0.7 * this.confEma;
    this.weakest = weakestPart(lms, need);
    this.buf[this.buf.length - 1].conf = conf;
    while (this.frameTimes.length && this.frameTimes[0] < keepFrom) this.frameTimes.shift();

    const low = conf < FRAME_MIN;
    const out: FrameOutput = {
      ...base, visible: true, primary: m.primary, confidence: r2(this.confEma), weakest: this.weakest, lowConfidence: low,
    };
    if (low) out.message = `Can't see your ${this.weakest} clearly`;
    // no form feedback from frames we don't trust
    const live = this.fsm.state === "idle" || low ? [] : def.liveChecks(m);
    out.badJoints = live.flatMap((f) => f.joints);
    let cue = live.length ? this.cues.offer(FAULT_INFO[live[0].code].cue, t, 2) : null;

    const ev = this.fsm.update(t, def.signal(m.primary));
    if (ev?.type === "rep") {
      const angleFrom = Math.max(this.lastRepEnd, ev.timing.start - 1.5);
      const features = aggregateRep(this.buf, ev.timing, def.concentricFirst, angleFrom);
      this.lastRepEnd = ev.timing.end;
      const { start, end } = ev.timing;
      const inWin = (x: number) => x >= start - 1e-6 && x <= end + 1e-6;
      const q = repQuality(
        this.buf.filter((b) => inWin(b.t)).map((b) => b.conf),
        this.frameTimes.filter(inWin).length,
        end - start,
        this.weakest,
      );
      const verdict = q.scored ? evaluateRep(this.exercise, features) : { score: null, faults: [] as FaultCode[] };
      this.setReps += 1;
      const rep: RepResult = {
        n: this.reps.length + 1, set: this.set, score: verdict.score, faults: verdict.faults,
        confidence: q.confidence, scored: q.scored, abstainReason: q.reason,
        eccS: features.eccS, conS: features.conS, rom: features.rom,
        t: r2(this.elapsed(t)), features,
      };
      this.reps.push(rep);
      out.rep = rep;
      if (!q.scored) cue = this.cues.offer("Rep counted, form not scored", t, 2) ?? cue;
      else cue = verdict.faults.length
        ? this.cues.offer(FAULT_INFO[verdict.faults[0]].cue, t, 3) ?? cue
        : (this.setReps % 3 === 0 ? this.cues.offer("Good reps, keep going", t, 1) : null) ?? cue;
    } else if (ev?.type === "partial" && !low) {
      const partialCue = { squat: "Go lower", pushup: "Chest to the floor", curl: "Full range, all the way up",
        press: "Press all the way up", lunge: "Drop the back knee" }[this.exercise];
      cue = this.cues.offer(partialCue, t, 3) ?? cue;
    }
    out.cue = cue?.text ?? null;
    out.state = this.fsm.state;
    out.setReps = this.setReps;
    out.totalReps = this.reps.length;
    return out;
  }

  /** Payload for POST /api/sessions/{id}/finish */
  toApiReps() {
    return this.reps.map((r) => ({
      set: r.set, score: r.score, faults: r.faults, ecc_s: r.eccS, con_s: r.conS,
      rom: r.rom, t: r.t, features: toApiFeatures(r.features),
      confidence: r.confidence, scored: r.scored, abstain_reason: r.abstainReason,
    }));
  }
}
