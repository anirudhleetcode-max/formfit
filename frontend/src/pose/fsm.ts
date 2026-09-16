// Rep counter: a finite-state machine over a smoothed 1-D "rep signal".
//
// The signal is a joint angle arranged so that the rest position is HIGH (e.g. knee angle for a
// squat: ~175 standing, ~70 at the bottom). A rep is:  top -> down -> bottom -> back to top.
//
//   idle ──(s ≥ top)──> top ──(s < top - hyst)──> down ──(s ≤ bottom)──> bottom
//                        ^                          │                      │
//                        └──── partial (s ≥ top) ───┘                      │
//                        └──────────────── rep (s ≥ top, long enough) ─────┘
//
// Two thresholds with a gap (hysteresis) stop noise around a single threshold from producing
// double counts, and a minimum rep duration rejects fast jitter.

export type FsmState = "idle" | "top" | "down" | "bottom";

export type FsmConfig = {
  top: number;          // signal at/above this = at the top (rest) position
  bottom: number;       // signal at/below this = deep enough to count
  hysteresis: number;   // must drop below top - hysteresis to leave the top
  minRepS: number;      // reps faster than this are treated as jitter
  maxRepS: number;      // a rep that takes longer than this is abandoned
  partialMinDrop: number; // a return to top after dropping at least this much (but not reaching bottom) = partial rep
};

export type RepTiming = {
  start: number;     // last time the signal was at the top before the rep (s)
  turn: number;      // time of the minimum (s)
  end: number;       // time the signal came back to the top (s)
  downS: number;     // start -> turn
  upS: number;       // turn -> end
  minSignal: number;
  maxSignal: number;
};

export type FsmEvent =
  | { type: "rep"; timing: RepTiming }
  | { type: "partial"; minSignal: number }
  | { type: "rejected"; reason: "too_fast" | "too_slow" }
  | { type: "started" };

export const DEFAULT_FSM: FsmConfig = {
  top: 155, bottom: 110, hysteresis: 8, minRepS: 0.6, maxRepS: 12, partialMinDrop: 20,
};

export class RepFsm {
  state: FsmState = "idle";
  count = 0;
  private cfg: FsmConfig;
  private lastTopT = 0;
  private minS = Infinity;
  private minT = 0;
  private maxS = -Infinity;

  constructor(cfg: Partial<FsmConfig> = {}) {
    this.cfg = { ...DEFAULT_FSM, ...cfg };
    if (this.cfg.bottom >= this.cfg.top - this.cfg.hysteresis) {
      throw new Error("bottom threshold must be below top - hysteresis");
    }
  }

  get config(): FsmConfig { return this.cfg; }

  reset() {
    this.state = "idle";
    this.count = 0;
    this.minS = Infinity;
    this.maxS = -Infinity;
  }

  /** Feed one smoothed sample. t in seconds. Returns an event when something happened. */
  update(t: number, s: number): FsmEvent | null {
    if (!Number.isFinite(s)) return null;
    const c = this.cfg;
    switch (this.state) {
      case "idle":
        if (s >= c.top) {
          this.state = "top";
          this.lastTopT = t;
          this.maxS = s;
          return { type: "started" };
        }
        return null;

      case "top":
        if (s >= c.top - c.hysteresis) {
          if (s >= c.top) this.lastTopT = t;
          this.maxS = Math.max(this.maxS, s);
          return null;
        }
        this.state = "down";
        this.minS = s;
        this.minT = t;
        return null;

      case "down":
      case "bottom": {
        if (s < this.minS) { this.minS = s; this.minT = t; }
        if (t - this.lastTopT > c.maxRepS) {
          this.state = "idle";
          return { type: "rejected", reason: "too_slow" };
        }
        if (this.state === "down" && s <= c.bottom) this.state = "bottom";
        if (s < c.top) return null;

        // back at the top
        const wasBottom = this.state === "bottom";
        const minS = this.minS;
        const timing: RepTiming = {
          start: this.lastTopT, turn: this.minT, end: t,
          downS: this.minT - this.lastTopT, upS: t - this.minT,
          minSignal: minS, maxSignal: Math.max(this.maxS, s),
        };
        this.state = "top";
        this.lastTopT = t;
        this.maxS = s;
        this.minS = Infinity;
        if (!wasBottom) {
          return timing.maxSignal - minS >= c.partialMinDrop ? { type: "partial", minSignal: minS } : null;
        }
        if (t - timing.start < c.minRepS) return { type: "rejected", reason: "too_fast" };
        this.count += 1;
        return { type: "rep", timing };
      }
    }
  }
}
