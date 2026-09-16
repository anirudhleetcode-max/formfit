import { Camera, Flag, Plus, Square, Volume2, VolumeX } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import ExercisePicker from "../components/ExercisePicker";
import { EMPTY_HUD, Stage, type HudState } from "../components/Stage";
import { useSpeech } from "../hooks/useSpeech";
import { api } from "../lib/api";
import { detect, drawSkeleton, getLandmarker, type Delegate } from "../lib/pose-runtime";
import type { Session } from "../lib/types";
import { PoseSession } from "../pose/engine";
import { EXERCISES, type ExerciseId } from "../pose/rules";

type Phase = "off" | "starting" | "ready" | "active" | "saving";

declare global {
  interface Window { __formfit?: { frames: number; poseFrames: number; reps: number; delegate?: string } }
}

export default function Train() {
  const nav = useNavigate();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number>(0);
  const engineRef = useRef<PoseSession | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const startRef = useRef(0);
  const hudTick = useRef(0);
  const lastVideoTime = useRef(-1);
  const cueAt = useRef(0);

  const [exercise, setExercise] = useState<ExerciseId>("squat");
  const [phase, setPhase] = useState<Phase>("off");
  const [hud, setHud] = useState<HudState>(EMPTY_HUD);
  const [error, setError] = useState<string | null>(null);
  const [delegate, setDelegate] = useState<Delegate | null>(null);
  const speech = useSpeech();
  const sayRef = useRef(speech.say);
  useEffect(() => { sayRef.current = speech.say; }, [speech.say]);

  const stopCamera = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const loop = useCallback(async () => {
    const { landmarker } = await getLandmarker();
    const tick = () => {
      const video = videoRef.current, canvas = canvasRef.current;
      if (!video || !canvas || !streamRef.current) return;
      if (video.readyState >= 2 && video.currentTime !== lastVideoTime.current) {
        lastVideoTime.current = video.currentTime;
        if (canvas.width !== video.videoWidth) { canvas.width = video.videoWidth; canvas.height = video.videoHeight; }
        const now = performance.now();
        let lms = null;
        try { lms = detect(landmarker, video, now); } catch (e) { console.error(e); }
        const stats = (window.__formfit ??= { frames: 0, poseFrames: 0, reps: 0 });
        stats.frames += 1;
        if (lms) stats.poseFrames += 1;

        const eng = engineRef.current;
        const out = eng ? eng.process(lms, now / 1000, video.videoWidth / Math.max(1, video.videoHeight)) : null;
        drawSkeleton(canvas.getContext("2d")!, lms, new Set(out?.badJoints ?? []));
        if (out?.cue) { sayRef.current(out.cue); cueAt.current = now; }
        if (eng) stats.reps = eng.reps.length;

        // re-render the HUD at most ~8x per second, or immediately on a new rep / cue
        if (out?.rep || out?.cue || now - hudTick.current > 120) {
          hudTick.current = now;
          setHud((h) => ({
            ...h,
            setReps: eng?.setReps ?? 0,
            set: eng?.set ?? h.set,
            totalReps: eng?.reps.length ?? 0,
            last: out?.rep ?? h.last,
            elapsed: eng ? (now - startRef.current) / 1000 : 0,
            cue: out?.cue ?? (now - cueAt.current < 2500 ? h.cue : null),
            tracking: lms && (out ? out.visible : true) ? "tracking" : "lost",
            message: out?.message ?? (lms ? null : "Looking for you"),
            confidence: out ? out.confidence : lms ? h.confidence : 0,
            lowConfidence: out ? out.lowConfidence : false,
          }));
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  async function startCamera() {
    setError(null);
    setPhase("starting");
    setHud({ ...EMPTY_HUD, tracking: "loading" });
    try {
      const [stream, loaded] = await Promise.all([
        navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false }),
        getLandmarker(),
      ]);
      streamRef.current = stream;
      setDelegate(loaded.delegate);
      (window.__formfit ??= { frames: 0, poseFrames: 0, reps: 0 }).delegate = loaded.delegate;
      const v = videoRef.current!;
      v.srcObject = stream;
      await v.play();
      setPhase("ready");
      setHud((h) => ({ ...h, tracking: "lost" }));
      loop();
    } catch (e) {
      stopCamera();
      setPhase("off");
      setHud(EMPTY_HUD);
      const name = (e as Error).name;
      setError(
        name === "NotAllowedError" ? "Camera permission was blocked. Allow it in the address bar, or use Upload instead."
          : name === "NotFoundError" ? "No camera found. You can analyse a recorded video in Upload."
          : `Could not start: ${(e as Error).message}`,
      );
    }
  }

  async function startSession() {
    setError(null);
    try {
      const s = await api<Session>("/api/sessions", { method: "POST", json: { exercise, source: "live" } });
      sessionIdRef.current = s.id;
      engineRef.current = new PoseSession(exercise);
      startRef.current = performance.now();
      cueAt.current = performance.now();
      setHud((h) => ({ ...h, setReps: 0, set: 1, totalReps: 0, last: null, cue: "Get in position" }));
      setPhase("active");
    } catch (e) {
      setError((e as Error).message);
    }
  }

  function nextSet() {
    engineRef.current?.nextSet();
    cueAt.current = performance.now();
    setHud((h) => ({ ...h, set: engineRef.current!.set, setReps: 0, cue: `Set ${engineRef.current!.set}` }));
  }

  async function finish() {
    const eng = engineRef.current, id = sessionIdRef.current;
    if (!eng || !id) return;
    setPhase("saving");
    try {
      await api(`/api/sessions/${id}/finish`, {
        method: "POST",
        json: { reps: eng.toApiReps(), duration_s: (performance.now() - startRef.current) / 1000 },
      });
      stopCamera();
      engineRef.current = null;
      nav(`/history/${id}`);
    } catch (e) {
      setError((e as Error).message);
      setPhase("active");
    }
  }

  const def = EXERCISES[exercise];
  const cameraOn = phase !== "off" && phase !== "starting";

  return (
    <div className="page train">
      <header className="page-head">
        <div>
          <h1>Train</h1>
          <p className="sub">{def.setup} Video is processed on this device and never uploaded.</p>
        </div>
        <button
          className="btn ghost"
          onClick={() => speech.setEnabled(!speech.enabled)}
          disabled={!speech.supported}
          aria-pressed={speech.enabled}
          title="Spoken cues"
        >
          {speech.enabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
          Voice {speech.enabled ? "on" : "off"}
        </button>
      </header>

      <div className="toolbar">
        <ExercisePicker value={exercise} onChange={setExercise} disabled={phase === "active" || phase === "saving"} />
        <div className="actions">
          {!cameraOn && (
            <button className="btn primary" onClick={startCamera} disabled={phase === "starting"} data-testid="start-camera">
              <Camera size={16} /> {phase === "starting" ? "Starting…" : "Start camera"}
            </button>
          )}
          {phase === "ready" && (
            <button className="btn primary" onClick={startSession} data-testid="start-session">
              <Flag size={16} /> Start session
            </button>
          )}
          {(phase === "active" || phase === "saving") && (
            <>
              <button className="btn" onClick={nextSet} disabled={phase === "saving" || hud.setReps === 0} data-testid="next-set">
                <Plus size={16} /> Next set
              </button>
              <button className="btn primary" onClick={finish} disabled={phase === "saving"} data-testid="finish">
                <Square size={14} /> {phase === "saving" ? "Saving…" : "End session"}
              </button>
            </>
          )}
        </div>
      </div>
      {error && <p className="alert" role="alert">{error}</p>}

      <Stage
        videoRef={videoRef}
        canvasRef={canvasRef}
        mirrored
        hud={hud}
        placeholder={!cameraOn ? (
          <div>
            <p className="big-hint">Set the camera about 2–3 m away</p>
            <p className="muted">{def.setup}</p>
          </div>
        ) : undefined}
      >
        <div className="hud-meta">
          {phase === "ready" && <p className="muted">Skeleton showing? Press <b>Start session</b> and begin your first set.</p>}
          {phase === "active" && hud.totalReps === 0 && <p className="muted">Stand tall to start. Reps count when you return to the top.</p>}
          {delegate && <p className="tiny muted">Pose model: lite · {delegate}</p>}
        </div>
      </Stage>
    </div>
  );
}
