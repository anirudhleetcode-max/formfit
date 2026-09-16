import { FileVideo, Play, Save } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import ExercisePicker from "../components/ExercisePicker";
import { EMPTY_HUD, Stage, type HudState } from "../components/Stage";
import { api } from "../lib/api";
import { detect, drawSkeleton, getLandmarker } from "../lib/pose-runtime";
import type { Session } from "../lib/types";
import { PoseSession } from "../pose/engine";
import { FAULT_INFO, type ExerciseId } from "../pose/rules";

const MAX_MB = 300;
const STEP_S = 1 / 15; // analyse at 15 frames per second of video, independent of CPU speed

type Phase = "empty" | "loaded" | "analysing" | "done" | "saving";

function seek(video: HTMLVideoElement, t: number) {
  return new Promise<void>((resolve) => {
    const done = () => { video.removeEventListener("seeked", done); resolve(); };
    video.addEventListener("seeked", done);
    video.currentTime = t;
  });
}

export default function Upload() {
  const nav = useNavigate();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const urlRef = useRef<string | null>(null);
  const cancelRef = useRef(false);
  const engineRef = useRef<PoseSession | null>(null);

  const [exercise, setExercise] = useState<ExerciseId>("squat");
  const [file, setFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<Phase>("empty");
  const [progress, setProgress] = useState(0);
  const [hud, setHud] = useState<HudState>(EMPTY_HUD);
  const [error, setError] = useState<string | null>(null);
  const [poseRate, setPoseRate] = useState<number | null>(null);
  const [reps, setReps] = useState<PoseSession["reps"]>([]);

  useEffect(() => () => {
    cancelRef.current = true;
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
  }, []);

  function pick(f: File | undefined) {
    setError(null);
    if (!f) return;
    if (!f.type.startsWith("video/")) { setError("That file is not a video."); return; }
    if (f.size > MAX_MB * 1024 * 1024) { setError(`Videos up to ${MAX_MB} MB, please trim longer recordings.`); return; }
    cancelRef.current = true;
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = URL.createObjectURL(f);
    const v = videoRef.current!;
    v.srcObject = null;
    v.src = urlRef.current;
    v.load();
    setFile(f);
    setPhase("loaded");
    setHud(EMPTY_HUD);
    setPoseRate(null);
    setReps([]);
    canvasRef.current?.getContext("2d")?.clearRect(0, 0, 9999, 9999);
  }

  async function analyse() {
    const video = videoRef.current!, canvas = canvasRef.current!;
    setError(null);
    setPhase("analysing");
    setHud({ ...EMPTY_HUD, tracking: "loading" });
    cancelRef.current = false;
    try {
      const { landmarker } = await getLandmarker();
      if (video.readyState < 1) await new Promise((r) => video.addEventListener("loadedmetadata", r, { once: true }));
      const duration = video.duration;
      if (!Number.isFinite(duration) || duration <= 0) throw new Error("Could not read the video length.");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const eng = new PoseSession(exercise);
      engineRef.current = eng;
      const ctx = canvas.getContext("2d")!;
      let poseFrames = 0, frames = 0, last = null as PoseSession["reps"][number] | null;
      for (let t = 0; t < duration; t += STEP_S) {
        if (cancelRef.current) return;
        await seek(video, t);
        const lms = detect(landmarker, video, t * 1000);
        frames += 1;
        if (lms) poseFrames += 1;
        const out = eng.process(lms, t, canvas.width / Math.max(1, canvas.height));
        if (out.rep) last = out.rep;
        drawSkeleton(ctx, lms, new Set(out.badJoints));
        setProgress(Math.min(1, (t + STEP_S) / duration));
        setHud((h) => ({
          ...h, setReps: eng.setReps, totalReps: eng.reps.length, last, elapsed: t,
          cue: out.cue ?? h.cue, tracking: out.visible ? "tracking" : "lost", message: out.message,
        }));
      }
      setPoseRate(frames ? poseFrames / frames : 0);
      setReps([...eng.reps]);
      window.__formfit = { frames, poseFrames, reps: eng.reps.length };
      setPhase("done");
    } catch (e) {
      setError((e as Error).message);
      setPhase("loaded");
    }
  }

  async function save() {
    const eng = engineRef.current;
    if (!eng) return;
    setPhase("saving");
    try {
      const s = await api<Session>("/api/sessions", { method: "POST", json: { exercise, source: "upload" } });
      await api(`/api/sessions/${s.id}/finish`, { method: "POST", json: { reps: eng.toApiReps(), duration_s: videoRef.current?.duration ?? 0 } });
      nav(`/history/${s.id}`);
    } catch (e) {
      setError((e as Error).message);
      setPhase("done");
    }
  }

  const busy = phase === "analysing" || phase === "saving";

  return (
    <div className="page upload">
      <header className="page-head">
        <div>
          <h1>Upload a set</h1>
          <p className="sub">Analyse a recorded video with the same pose pipeline. The file stays in this browser.</p>
        </div>
      </header>

      <div className="toolbar">
        <ExercisePicker value={exercise} onChange={setExercise} disabled={busy} />
        <div className="actions">
          <label className={"btn" + (busy ? " disabled" : "")}>
            <FileVideo size={16} /> {file ? "Change video" : "Choose video"}
            <input type="file" accept="video/*" hidden disabled={busy} data-testid="video-input"
              onChange={(e) => pick(e.target.files?.[0])} />
          </label>
          {(phase === "loaded" || phase === "done") && (
            <button className="btn primary" onClick={analyse} data-testid="analyse">
              <Play size={16} /> {phase === "done" ? "Analyse again" : "Analyse"}
            </button>
          )}
          {(phase === "done" || phase === "saving") && (
            <button className="btn primary" onClick={save} disabled={phase === "saving" || reps.length === 0} data-testid="save-upload">
              <Save size={16} /> {phase === "saving" ? "Saving…" : "Save session"}
            </button>
          )}
        </div>
      </div>
      {error && <p className="alert" role="alert">{error}</p>}
      {file && (
        <div className="progress-line">
          <span className="file-name">{file.name}</span>
          <span className="bar" aria-hidden="true"><i style={{ width: `${progress * 100}%` }} /></span>
          <span className="num">{Math.round(progress * 100)}%</span>
        </div>
      )}

      <Stage
        videoRef={videoRef}
        canvasRef={canvasRef}
        mirrored={false}
        hud={hud}
        placeholder={!file ? (
          <div>
            <p className="big-hint">Drop in a clip of one exercise</p>
            <p className="muted">Side-on, whole body in frame. MP4 or WebM, up to {MAX_MB} MB.</p>
          </div>
        ) : undefined}
      >
        {phase === "done" && (
          <div className="hud-meta" data-testid="upload-result">
            <p><b className="num">{reps.length}</b> reps found · pose detected in <b className="num">{Math.round((poseRate ?? 0) * 100)}%</b> of frames</p>
            {reps.length === 0 && <p className="muted">No full reps. Check the exercise matches the video and the whole body is visible.</p>}
            <ol className="rep-mini">
              {reps.slice(0, 12).map((r) => (
                <li key={r.n}>
                  <span className="num">#{r.n}</span>
                  <span className="num">{r.score}</span>
                  <span className="muted">{r.faults.length ? r.faults.map((f) => FAULT_INFO[f].label).join(", ") : "clean"}</span>
                </li>
              ))}
            </ol>
          </div>
        )}
      </Stage>
    </div>
  );
}
