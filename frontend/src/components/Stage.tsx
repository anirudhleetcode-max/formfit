import type { ReactNode, RefObject } from "react";
import { FAULT_INFO } from "../pose/rules";
import type { RepResult } from "../pose/engine";
import { clock, scoreClass } from "../lib/format";

export type HudState = {
  setReps: number;
  set: number;
  totalReps: number;
  last: RepResult | null;
  elapsed: number;
  cue: string | null;
  tracking: "idle" | "loading" | "tracking" | "lost";
  message: string | null;
};

export const EMPTY_HUD: HudState = {
  setReps: 0, set: 1, totalReps: 0, last: null, elapsed: 0, cue: null, tracking: "idle", message: null,
};

export function Stage({ videoRef, canvasRef, mirrored, hud, placeholder, children }: {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  mirrored: boolean;
  hud: HudState;
  placeholder?: ReactNode;
  children?: ReactNode;
}) {
  const statusText = {
    idle: "Camera off", loading: "Loading pose model…", tracking: "Tracking", lost: hud.message ?? "Looking for you",
  }[hud.tracking];
  return (
    <div className="stage-wrap">
      <div className={"stage" + (mirrored ? " mirrored" : "")}>
        <video ref={videoRef} playsInline muted />
        <canvas ref={canvasRef} data-testid="overlay" />
        {placeholder && <div className="stage-empty">{placeholder}</div>}
        <div className={"track-pill t-" + hud.tracking} data-testid="tracking">
          <i aria-hidden="true" />{statusText}
        </div>
        {hud.cue && <div className="cue" role="status" aria-live="polite" key={hud.cue}>{hud.cue}</div>}
      </div>
      <aside className="hud" aria-label="Session stats">
        <div className="hud-reps">
          <span className="hud-label">Reps · set {hud.set}</span>
          <span className="hud-big" data-testid="rep-count">{hud.setReps}</span>
        </div>
        <dl className="hud-grid">
          <div><dt>Total</dt><dd data-testid="total-reps">{hud.totalReps}</dd></div>
          <div><dt>Time</dt><dd>{clock(hud.elapsed)}</dd></div>
          <div>
            <dt>Last rep</dt>
            <dd className={scoreClass(hud.last?.score)} data-testid="last-score">{hud.last ? hud.last.score : "—"}</dd>
          </div>
          <div>
            <dt>Tempo</dt>
            <dd className="hud-tempo">{hud.last ? `${hud.last.eccS.toFixed(1)}↓ ${hud.last.conS.toFixed(1)}↑` : "—"}</dd>
          </div>
        </dl>
        <div className="hud-faults">
          {hud.last && hud.last.faults.length === 0 && <span className="ok-line">Clean rep</span>}
          {hud.last?.faults.map((f) => <span key={f} className="fault-chip">{FAULT_INFO[f].label}</span>)}
        </div>
        {children}
      </aside>
    </div>
  );
}
