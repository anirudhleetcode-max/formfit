# FormFit — webcam rep counter and form coach

FormFit watches you train through the webcam, counts reps and flags form faults as they happen:
squats, push-ups, bicep curls, shoulder presses and lunges. **Video never leaves the browser.** Pose
estimation runs on-device with MediaPipe. Only the per-rep numbers (score, faults, tempo, range
of motion and 12 measured features) are sent to the API. The server stores sessions, detects
fatigue, scores every rep with a trained classifier and powers a progress dashboard.

Stack: React + Vite + TypeScript · FastAPI · MongoDB (Motor) · MediaPipe Tasks (browser) · scikit-learn.

| Live session | Session detail |
|---|---|
| ![live](docs/screenshots/01-live-session.png) | ![detail](docs/screenshots/02-session-detail.png) |
| **Progress** | **History** |
| ![progress](docs/screenshots/03-progress.png) | ![history](docs/screenshots/04-history.png) |

## Features

- **Train (live):** pick an exercise, start the camera, and a skeleton is drawn over the mirrored video. Joints turn red
  while a form rule is broken. The HUD shows reps in the current set (big enough to read from 2 m),
  the set number, total reps, time, the last rep's score and its tempo (down / up seconds). Short coaching cues
  ("Chest up", "Go lower") appear on screen and can be spoken aloud (Web Speech API toggle). Cues are throttled.
  Next set / End session buttons control the session.
- **Visibility checks:** reps are not counted while the needed joints are hidden or out of frame
  ("Step back so your whole body is visible").
- **Upload:** analyse a recorded clip with the same pipeline, deterministically at 15 analysed frames per
  second of video, then save it as a session. Try `samples/squat_demo.webm`.
- **History:** paginated session list (cursor-based) with an exercise filter and a fatigue flag. The detail
  view has a rep-by-rep chart (rule score, model score, lift time, set boundaries, fatigue marker), a set
  table and a rep table with faults.
- **Progress:** weekly volume, daily form-score trend (rule and model), recurring faults, personal bests
  with a trend slope in points per week, and the model's evaluation metrics.
- **Demo account:** `demo@formfit.app` / `demo1234` (30 days of seeded data). The login page has a "Use demo account" link.

## Architecture

```
 Browser (all video stays here)                                   Server
┌───────────────────────────────────────────────────────┐      ┌──────────────────────────────────────┐
│ <video> webcam / uploaded file                        │      │ FastAPI  (app/)                      │
│    │ frame                                            │      │  auth.py      JWT + bcrypt           │
│    ▼                                                  │      │  routers/sessions.py  create/finish/ │
│ MediaPipe PoseLandmarker (lite, WASM, GPU→CPU)        │      │               list/detail/delete     │
│    │ 33 landmarks                                     │      │  routers/stats.py     $facet stats   │
│    ▼                                                  │ JSON │  services/form_model.py  sklearn GB  │
│ pose/engine.ts  PoseSession                           │ reps │  services/analytics.py   fatigue,    │
│   rules.ts  frame metrics (angles, lean, valgus…)     │─────▶│                          trends      │
│   angles.ts One Euro smoothing                        │      │  services/sessions.py  summaries     │
│   fsm.ts    rep state machine (hysteresis, min time)  │      └──────────────┬───────────────────────┘
│   rules.ts  live checks → red joints + cues           │                     │ Motor
│             per-rep features → score + faults         │              ┌──────▼──────┐
│ canvas overlay · HUD · speech                         │              │  MongoDB    │ users, sessions
└───────────────────────────────────────────────────────┘              └─────────────┘
 Offline (python -m ...): ml/simulate.py → ml/train.py → ml/artifacts/form_model.joblib + metrics.json
```

### Rep pipeline (frontend/src/pose, pure TypeScript, unit-tested)

1. **Landmarks → frame metrics** (`rules.ts: frameMetrics`). x is rescaled by the video aspect ratio first,
   because MediaPipe normalises x by width and y by height. Metrics: primary joint angle (knee for
   squat/lunge, elbow for push-up/curl/press), torso lean from vertical, shoulder-hip-ankle line and signed hip
   offset (sag vs pike), knee/ankle width ratio (valgus, only in a frontal view), upper-arm-to-torso angle
   (elbow drift), hip-below-knee depth, and heel lift (side view with the foot in frame only). The view is
   classified from shoulder width / torso length (side < 0.35 < oblique < 0.45 < frontal).
2. **Smoothing** with a One Euro filter (adaptive low-pass: smooth when still, low lag when moving).
3. **Rep FSM** (`fsm.ts`): `idle → top → down → bottom → top`. Two thresholds with a hysteresis gap stop
   noise from double counting. Reps shorter than a minimum duration are rejected as jitter, stalled reps
   are abandoned, and a return to the top without reaching the bottom is a *partial* ("Go lower").
   The press uses `180 − elbow angle` so its rest position (the rack) is "high" like every other exercise.
   Tempo = time from leaving the top to the turning point, and from there back to the top.
4. **Rep features → rule score** (`rules.ts: evaluate`): 17 fault rules with fixed penalties
   (score = 100 − Σ penalties). Examples: squat depth (hip more than 0.15 thigh-lengths above the knee), torso
   lean > 45°, knee valgus < 0.8, heel lift, descent < 0.5 s, no lockout; push-up hip line < 160°, elbow
   > 95° at the bottom; curl elbow drift > 28°, torso sway > 11°, partial range; press lockout < 158°, back arch.

## ML approach (server)

### Form-quality classifier (`backend/ml/`)
- **Data: synthetic, and this matters.** There is no public dataset of per-rep pose features with expert
  labels, so `ml/simulate.py` generates reps from a simplified biomechanical model:
  - Subjects are sampled with Winter anthropometric segment ratios (with noise), ankle mobility and a camera view.
  - True rep parameters come from clean / fault / borderline mixtures, so the classes overlap.
  - For the squat, torso lean is *solved* from a 2-D sagittal balance model: the whole-body centre of mass
    must stay over mid-foot given the shank tilt that ankle mobility allows. Long femurs and stiff ankles
    therefore force more lean.
  - A 30 fps angle trajectory gets view-dependent landmark jitter, and features are measured the way the
    browser measures them (with projection losses when not side-on).
  - Label = clean if no ground-truth fault holds on the **true** parameters. The model only sees the noisy
    measurements.
- **Models:** one model per exercise, using only that exercise's relevant features (so measurements that
  don't apply, like elbow flare under a barbell, can't push real reps out of distribution). Logistic
  regression, random forest and gradient boosting are compared on a validation split, and the best by ROC-AUC is
  refit with balanced sample weights. Gradient boosting won for every exercise.
- **Serving:** loaded once at startup (lifespan) and run with `run_in_threadpool` when a session is finished.
  Each rep gets `model_score = 100 · P(clean)` next to the rule score.
- **Results** (`backend/ml/artifacts/metrics.json`). Test set: 2,000 independently generated reps per exercise; 6,000 training reps.

| Exercise | Clean rate | Model acc. | Model F1 (faulty) | Model ROC-AUC | Rules acc. | Rules ROC-AUC |
|---|---|---|---|---|---|---|
| Squat | 0.27 | 0.880 | 0.912 | 0.952 | 0.834 | 0.866 |
| Push-up | 0.49 | 0.979 | 0.979 | 0.998 | 0.967 | 0.968 |
| Bicep curl | 0.33 | 0.931 | 0.946 | 0.979 | 0.907 | 0.928 |
| Shoulder press | 0.57 | 0.919 | 0.903 | 0.974 | 0.914 | 0.901 |
| Lunge | 0.48 | 0.894 | 0.890 | 0.952 | 0.863 | 0.870 |

The numbers show the model learns this simulator well. They do **not** show real-world accuracy, which is unknown
until the model is validated on labelled real reps. On the one real clip in `samples/`, the 4 demonstration
squats get rule score 100 and model scores 72–81.

The two scores use different scales: the rule score takes 15–30 points off per fault, while the model score is a
probability, so a rep with one clear fault gets roughly 70–85 from the rules but under 20 from the model.

### Fatigue detection (`app/services/analytics.py`)
- A per-rep **fatigue index** averages three relative changes against the median of the first 3 reps:
  slower concentric (lifting) time, smaller ROM and lower score.
- A **single change-point** (least-squares mean shift) is fitted to that index. Fatigue is flagged when the
  shift is ≥ 0.12, its Welch t-statistic is ≥ 3, and the overall linear slope is positive. The onset rep is stored.
- Per-metric OLS slopes (s/rep, °/rep, points/rep) are also stored.
- Evaluation (`python -m ml.eval_fatigue`, 300 simulated 12-rep squat sets per pattern):

| Pattern | Flagged |
|---|---|
| no fatigue (false-positive rate) | 5.7% |
| mild progressive | 25.7% |
| strong progressive | 57.7% |
| sudden drop after rep 8 | 80.7% |

The detector is deliberately conservative: false alarms are more annoying than a missed mild slowdown.

### Progress analytics
MongoDB `$facet` aggregation computes weekly volume (`$dateTrunc` in the user's time zone), daily average scores,
fault counts (`$unwind` reps → faults) and personal bests. A per-exercise least-squares slope turns the daily
scores into "points per week".

## API

All endpoints except auth and health need `Authorization: Bearer <token>`. Every session query filters by `user_id`.
Errors are always `{"detail": ...}`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/auth/register` | create account → token |
| POST | `/api/auth/login` | sign in → token |
| GET | `/api/auth/me` | current user |
| GET | `/api/health` | DB + model status |
| POST | `/api/sessions` | start a session `{exercise, source}` |
| POST | `/api/sessions/{id}/finish` | `{reps[], duration_s}` → model scores, set summary, fatigue (409 if already finished) |
| GET | `/api/sessions?limit=&cursor=&exercise=` | finished sessions, newest first, cursor pagination (no rep arrays) |
| GET | `/api/sessions/{id}` | full session with reps |
| DELETE | `/api/sessions/{id}` | delete |
| GET | `/api/stats/overview?weeks=&tz=` | totals, weekly volume, daily scores, faults, bests + trends |
| GET | `/api/model` | classifier status and evaluation metrics |

- **Validation:** exercise enum, known fault codes only, unknown feature keys dropped, score 0–100, at most 500 reps,
  and a 1 MB request body limit (413).
- **Indexes:** `users.email` (unique), `sessions(user_id, status, started_at, _id)`, `sessions(user_id, exercise, started_at)`.

## Setup on Windows

Prerequisites: Python 3.11+, Node 20+, and MongoDB Community running on `mongodb://127.0.0.1:27017`.
No Tesseract or GPU is needed; any webcam works.

```powershell
# 1. backend
cd formfit\backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env            # then set JWT_SECRET to a long random string
python -m ml.train                # ~40 s, (re)creates ml\artifacts\form_model.joblib
python -m scripts.seed            # demo@formfit.app / demo1234
uvicorn app.main:app --reload --port 8003

# 2. frontend (second terminal)
cd formfit\frontend
npm install
npm run dev                       # http://localhost:5175
```

- `npm run dev` and `npm run build` first run `scripts/copy-wasm.mjs`, which copies the MediaPipe WASM runtime
  from `node_modules` into `public/mediapipe/wasm` (no CDN at runtime). They also run `scripts/fetch-model.mjs`,
  which downloads `pose_landmarker_lite.task` (5.6 MB) into `public/models/` if it isn't there. The model file is
  small, so it is kept in the repo and the app works offline. `npm run fetch-model` re-downloads it after deletion.
- The trained `form_model.joblib` (0.3 MB) is committed, but joblib files are tied to the scikit-learn version
  (pinned `>=1.8,<1.9`). If `/api/health` shows `"model": false`, run `python -m ml.train`.
- Browsers only allow the camera on `localhost` or HTTPS. Use `http://localhost:5175`, not your LAN IP.

## Tests

```bash
cd backend && python -m pytest -q          # 18 tests: API (real Mongo, throwaway DB) + ML/analytics units
cd frontend && npm test                    # 28 vitest tests: angles, One Euro, FSM, rules, cues,
                                           # engine on synthetic skeletons + REAL recorded landmarks
cd e2e && ./run_e2e.sh                     # Playwright journey (bash; on Windows use Git Bash or WSL)
```

- The frontend real-clip test replays MediaPipe landmarks recorded from `samples/squat_demo.webm`
  (`python -m scripts.dump_landmarks`) and asserts that all 4 squats are counted and scored clean.
- The e2e runner:
  - converts the sample clip to `.y4m`, and Chromium plays it as a fake webcam
    (`--use-fake-device-for-media-stream --use-file-for-fake-video-capture`);
  - starts the backend on 8003 and `vite preview` on 5175;
  - registers a user, runs a live squat session (asserts landmarks, a painted skeleton and ≥ 2 reps),
    ends it, and checks the detail page and chart;
  - analyses the same clip in Upload mode (asserts pose found in > 80% of frames and 3–5 reps; it finds 4)
    and saves it;
  - checks History and Progress, then logs in as the demo user for the dashboard screenshots
    and a 375 px no-overflow check;
  - removes its throwaway users at the end.

  The headless run uses `?delegate=cpu` because the GPU delegate on SwiftShader works but is slow. Normal browsers try GPU first and fall back to CPU.

**Sample video:** "Squat - exercise demonstration video" by FitnessScape, CC BY 3.0, via Wikimedia Commons
(see `samples/ATTRIBUTION.md`).

## Viva notes

- **Why run pose in the browser?** Privacy (frames never leave the device), no GPU server cost, and no
  network round-trip per frame. The server only receives about 20 numbers per rep.
- **Why a state machine rather than peak counting?** It works online (no look-ahead). Hysteresis kills
  threshold chatter, and it gives natural hooks for partial reps, stalls and tempo.
- **Why One Euro instead of a moving average?** A fixed EMA trades lag for smoothness everywhere. One Euro raises its
  cutoff when the signal moves fast, so the bottom of a rep isn't flattened and the standing position still doesn't jitter.
- **Why rescale x by the aspect ratio?** Normalised coordinates distort angles on 16:9 video. That bug was found on
  the real clip, where a rear view was read as side-on and produced false heel-lift faults.
- **Why both rules and a model?** Rules are explainable and drive the live cues. The model combines
  features non-linearly and, on the simulator, beats the rules on ROC-AUC for every exercise (e.g. squat 0.95 vs 0.87).
- **How is the model not "fake AI"?** It is trained and evaluated by reproducible scripts, and the numbers above are
  what those scripts printed. Its limit is the synthetic data, and that is stated plainly.
- **How does fatigue detection work?** A change-point on a composite fatigue index with a t-test gate.
  False-positive rate is about 6% on simulated fatigue-free sets.
- **Scaling:** stats are one aggregation round-trip, lists use cursor pagination on an index, rep arrays are
  excluded from lists, and model inference runs in a thread pool.
- **Likely question: "What if the user is side-on for squats but facing the camera for presses?"** Each exercise
  says which joints must be visible and whether one side is enough. The view classifier turns off checks that
  can't be measured from that angle (valgus needs a frontal view, heel lift needs a side view).

## Limitations / future work

- The classifier is trained on simulated reps only. Next steps: collect labelled real reps (the app already stores
  the feature vectors) and fine-tune or recalibrate on them.
- The 2-D pose from one camera loses depth. Knee valgus and torso lean are view-dependent approximations.
- Headless live-camera e2e processes only a few frames per second on this CI box. Real browsers with a GPU run at 20–30 fps.
- Single person only (`numPoses: 1`). Alternating curls count the more visible arm.
- Possible additions: per-user thresholds (mobility differs), rep velocity with a calibration object, and PWA/offline session queueing.
