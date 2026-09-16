import { useEffect, useState } from "react";
import { DISCLAIMER, Skeleton } from "../components/Feedback";
import { api } from "../lib/api";
import { exLabel } from "../lib/format";
import type { ModelInfo } from "../lib/types";
import { FRAME_MIN, MIN_REP_FPS, REP_MIN } from "../pose/confidence";
import type { ExerciseId } from "../pose/rules";

const fmt = (x: number | null | undefined, d = 3) => (x === null || x === undefined ? "—" : x.toFixed(d));

const PIPELINE: [string, string][] = [
  ["Frame", "Webcam or uploaded video frame. Stays in the browser; nothing is uploaded."],
  ["Pose model", "MediaPipe PoseLandmarker lite (pretrained by Google, not trained here). 33 landmarks with a visibility score. GPU delegate, CPU fallback."],
  ["Geometry", "x is rescaled by the video aspect ratio, then joint angles, torso lean, hip line, knee/ankle width ratio, depth and heel lift are measured. Camera view (side / oblique / front) switches off checks that cannot be measured from that angle."],
  ["Smoothing", "One Euro filter per signal (min cutoff 1.2 Hz, beta 0.015, derivative cutoff 1 Hz)."],
  ["Rep segmentation", "Finite-state machine on the main joint angle: top → down → bottom → top, with a hysteresis gap, a minimum rep time and partial-rep detection."],
  ["Tracking confidence", `Per frame: landmark visibility × jitter score. Below ${FRAME_MIN} no live cues are given. Per rep: mean frame confidence × dropout factor × frame-rate factor; below ${REP_MIN} (or under ${MIN_REP_FPS} analysed frames/s) the rep is counted but its form is not scored.`],
  ["Rule score", "17 form rules with fixed penalties, 100 − Σ penalties. Drives the live cues."],
  ["Model score (server)", "A tree ensemble per exercise (scikit-learn gradient boosting; random forest for lunges, picked on a validation split) estimates the probability that the rep is clean. Trained on synthetic reps only; accuracy on real lifters is unknown. Reported as uncertain below a validated confidence, and withheld when a feature is outside the training range."],
  ["Analytics (server)", "Fatigue: change-point on a per-rep index of lift time, range of motion and score, with a t-test gate. Progress: MongoDB aggregations."],
];

export default function About() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api<ModelInfo>("/api/model").then(setInfo).catch((e) => setError(e.message)); }, []);
  const card = info?.card;

  return (
    <div className="page about">
      <header className="page-head">
        <div>
          <h1>How FormFit works</h1>
          <p className="sub">What is measured, what the scores mean, how sure they are, and where they stop being useful.</p>
        </div>
      </header>

      <p className="disclaimer" role="note" data-testid="disclaimer">{DISCLAIMER}</p>

      <section className="panel">
        <h2>Data you send</h2>
        <p className="muted">
          Video frames never leave your browser. When a session ends, only per-rep numbers are saved: score, faults,
          tempo, range of motion, tracking confidence and 12 measured angles/ratios. Delete a session from its page at any time.
        </p>
      </section>

      <section className="panel">
        <h2>Pipeline</h2>
        <ol className="pipeline">
          {PIPELINE.map(([k, v], i) => (
            <li key={k}><span className="step num">{i + 1}</span><div><b>{k}</b><p className="muted">{v}</p></div></li>
          ))}
        </ol>
      </section>

      <section className="panel" aria-labelledby="model-h" data-testid="model-panel">
        <h2 id="model-h">Model card</h2>
        {error && <p className="alert">{error}</p>}
        {!info && !error && <Skeleton rows={4} />}
        {info && !info.ready && (
          <p className="alert" role="alert">The form classifier is not loaded ({info.error}). Rule scores still work; model scores are skipped.</p>
        )}
        {info?.ready && (
          <>
            <p className="synthetic-note">
              <b>Trained on synthetic data.</b> {info.training_data}
            </p>
            {card && (
              <dl className="card-grid">
                <div><dt>Version</dt><dd>{card.model_version}</dd></div>
                <div><dt>Trained</dt><dd>{card.trained_at?.slice(0, 10)}</dd></div>
                <div><dt>Dataset</dt><dd>{card.dataset.name} (simulator v{card.dataset.simulator_version})</dd></div>
                <div><dt>Size</dt><dd>{Object.entries(card.dataset.sizes).map(([k, v]) => `${k} ${v}`).join(" / ")} reps per exercise</dd></div>
                <div><dt>Features</dt><dd>{card.preprocessing_version}</dd></div>
                <div><dt>Seed</dt><dd>{card.seed}</dd></div>
                <div><dt>scikit-learn</dt><dd>{card.libraries?.scikit_learn}</dd></div>
                <div><dt>Commit</dt><dd>{card.git_commit ?? "—"}</dd></div>
              </dl>
            )}
            {info.metrics && (
              <>
                <h3>Held-out synthetic test set ({info.metrics.n_test?.toLocaleString()} reps per exercise)</h3>
                <div className="table-scroll">
                  <table className="table compact">
                    <thead>
                      <tr>
                        <th scope="col">Exercise</th><th scope="col">Model</th>
                        <th scope="col" className="r">ROC-AUC</th><th scope="col" className="r">F1 faulty</th>
                        <th scope="col" className="r">ECE</th><th scope="col" className="r">Rules AUC</th>
                        <th scope="col" className="r">Log. reg. AUC</th><th scope="col" className="r">Majority acc.</th>
                        <th scope="col" className="r">Confident coverage</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(info.metrics.exercises).map(([k, v]) => (
                        <tr key={k}>
                          <td>{exLabel(k as ExerciseId)}</td>
                          <td className="muted">{v.selected.replace("_", " ")}</td>
                          <td className="r num">{fmt(v.model.roc_auc)}</td>
                          <td className="r num">{fmt(v.model.f1_faulty)}</td>
                          <td className="r num">{fmt(v.model.ece)}</td>
                          <td className="r num muted">{fmt(v.rule_baseline.roc_auc)}</td>
                          <td className="r num muted">{fmt(v.logreg?.roc_auc)}</td>
                          <td className="r num muted">{fmt(v.majority?.accuracy)}</td>
                          <td className="r num">{v.selective ? `${Math.round(v.selective.coverage * 100)}%` : "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="tiny muted">
                  ECE = expected calibration error of P(clean). Confident coverage = share of test reps where the model was
                  confident enough to reach 95% accuracy on validation. These numbers describe the simulator, not real lifters.
                </p>
              </>
            )}
            {card && (
              <>
                <h3>Intended use and limits</h3>
                <p className="muted">{card.intended_use}</p>
                <ul className="bullets">{card.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
              </>
            )}
          </>
        )}
      </section>

      <section className="panel">
        <h2>Checked on real video</h2>
        <p className="muted">
          Rep counting was run on a small set of public Wikimedia Commons clips with hand-counted reps (see
          <code> experiments/</code> in the repository for the manifest, method and per-clip results). The sample is tiny,
          so treat it as a sanity check rather than an accuracy figure. Form scores and model scores have not been checked
          against coach-labelled real reps: <b>not measured yet</b>.
        </p>
      </section>
    </div>
  );
}
