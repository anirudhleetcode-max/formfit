// Cue throttling: at most one cue every `gapS` seconds, and the same cue is not repeated
// within `repeatS` seconds. Higher priority cues can interrupt after `urgentGapS`.

export type Cue = { text: string; priority: number; t: number };

export class CueThrottle {
  private last: Cue | null = null;
  private lastByText = new Map<string, number>();
  private gapS: number;
  private repeatS: number;
  private urgentGapS: number;
  constructor(gapS = 1.8, repeatS = 4, urgentGapS = 0.8) {
    this.gapS = gapS; this.repeatS = repeatS; this.urgentGapS = urgentGapS;
  }

  /** Offer a cue at time t (seconds). Returns the cue if it should be shown/spoken now. */
  offer(text: string, t: number, priority = 1): Cue | null {
    const prevSame = this.lastByText.get(text);
    if (prevSame !== undefined && t - prevSame < this.repeatS) return null;
    if (this.last) {
      const since = t - this.last.t;
      const gap = priority > this.last.priority ? this.urgentGapS : this.gapS;
      if (since < gap) return null;
    }
    const cue = { text, priority, t };
    this.last = cue;
    this.lastByText.set(text, t);
    return cue;
  }

  reset() {
    this.last = null;
    this.lastByText.clear();
  }
}
