# FormFit audit (phase 2, before changes)

Scope: every file in `backend/`, `frontend/src/`, `e2e/` and the README as of the baseline commit
`b76b986`. Baseline status: `npm run build` OK, vitest 28 passed, pytest 18 passed, e2e 2 passed.

Severity: **H** = misleading or unsafe, **M** = real gap a reviewer would notice, **L** = polish.

## 1. Broken or incomplete features

| # | Sev | Finding |
|---|---|---|
| F1 | M | A live session is created on "Start session". If the tab is closed, the `active` session stays in MongoDB forever (not listed, never cleaned up). |
| F2 | H | Reps are scored whenever the required joints pass a single visibility threshold (0.5). There is no notion of tracking quality, so a rep tracked through heavy jitter or dropouts still gets a confident score and faults. |
| F3 | M | Live cues fire even on frames where tracking is poor, so noise can produce "Chest up" when the person is fine. |
| F4 | M | No explicit "can't see X, form not scored" state; the only visibility message is a generic "Step back". |
| F5 | L | Delete uses `window.confirm`; there are no toasts, and loading states are plain "Loading…" text. |
| F6 | L | `ml/simulate.py` has an unused constant `G`. |

## 2. Duplicated code

| # | Sev | Finding |
|---|---|---|
| D1 | L | Rep rules exist twice: `frontend/src/pose/rules.ts` (source of truth) and `backend/ml/rules.py` (mirror for the baseline and the seed). A pytest checks the thresholds stay in sync. The Python mirror cannot tell a hip pike from a sag (it has no hip-offset feature). Acceptable, but it must be documented. |

## 3. Placeholder / demo / synthetic data and where it surfaces

| # | Sev | Finding |
|---|---|---|
| S1 | H | The form classifier is trained **only on simulated reps**. The UI calls its output "Model: clean-rep chance" / "Model %" without saying synthetic anywhere except one paragraph on Progress. |
| S2 | M | The demo account's 30 days of sessions are simulated (seed script), but History/Progress show them like real activity with no demo label. |
| S3 | M | The README's metrics table is synthetic-only; the only real-video check is one clip inside a unit test. |

## 4. Weak ML

| # | Sev | Finding |
|---|---|---|
| M1 | H | Evaluation is synthetic-only. There is no real-video evaluation harness or manifest. |
| M2 | H | The model is fit with **balanced sample weights**, so `model_score` is not a calibrated probability of anything. No calibration check (ECE / Brier / reliability table). |
| M3 | M | Baselines are incomplete: no majority-class baseline; the rule baseline has no threshold tuned on validation; the logistic regression is only used for model selection, not reported. |
| M4 | M | No confusion matrix, per-class report, PR-AUC, or threshold selection on a validation split. |
| M5 | M | No error analysis (worst false positives / false negatives). |
| M6 | H | No abstention: every rep gets a model score, even when its features are far outside the simulator's range (real reps are out of distribution by construction) or when tracking was poor. |
| M7 | M | The fatigue detector thresholds (shift 0.12, t 3.0) were tuned by looking at the same simulated sets (seed 3) that the reported false-alarm rate comes from. Tuning and evaluation seeds must differ. |
| M8 | M | No model card / sidecar metadata (dataset hash, seed, library versions, hyper-parameters) next to `form_model.joblib`. |
| M9 | L | The pipeline stages are spread over files without a document that maps stage → file → parameters. |

## 5. Security

| # | Sev | Finding |
|---|---|---|
| X1 | H | `JWT_SECRET` defaults to `change-me-in-production` and the app starts silently with it. |
| X2 | M | No login rate limiting (password guessing). |
| X3 | M | The body size limit relies on `Content-Length`; a chunked request bypasses it. |
| X4 | L | Logging is not configured (no format, no level); errors log the path only. Nothing sensitive is logged today, which should be kept. |
| X5 | L | `/api/health` returns 200 with `model: false` when the classifier failed to load; the degraded state is easy to miss. |
| X6 | — | Uploads: video never reaches the server (browser-only analysis), so there is no server-side file handling, path traversal or magic-byte surface. The client checks MIME type and size. The `features` dict is whitelisted, faults are an enum. OK. |
| X7 | — | CORS is restricted to configured origins. OK. |

## 6. UI / UX gaps

| # | Sev | Finding |
|---|---|---|
| U1 | H | No health/medical disclaimer anywhere. |
| U2 | M | No tracking-confidence indicator in the HUD or session detail. |
| U3 | M | No Model/About panel with a model card, the pipeline, and its limits. |
| U4 | L | Destructive delete has no styled confirm; there are no toasts. |
| U5 | L | Loading states are text only (no skeletons). |

## 7. Test gaps

| # | Sev | Finding |
|---|---|---|
| T1 | M | No React component tests (vitest covers only the pure pose modules). |
| T2 | M | No robustness tests: missing model file, corrupt model file, empty session, huge payload (without Content-Length), low-confidence reps, NaN features. |
| T3 | L | No test of the fatigue false-alarm rate. |
| T4 | L | No CI. |

## Prioritised plan

1. Tracking confidence + abstention in the TS pipeline (F2–F4, U2, M6); API accepts unscored reps (backward compatible).
2. Real-video evaluation: clip manifest, landmark extraction, a Node runner on the same TS engine, and counted ground truth (M1, S3).
3. Classifier evaluation rework: unweighted fit, calibration, baselines, threshold on validation, OOD check, error dump, model card, experiments folder (M2–M5, M8).
4. Fatigue: separate tuning and evaluation seeds, document, test (M7, T3).
5. Backend hardening: JWT secret policy, login rate limit, streaming body limit, logging, health degraded state, cleanup of stale active sessions (X1–X5, F1).
6. UI: disclaimer, About/Model page, synthetic + demo labels, confirm dialog, toasts, skeletons (U1–U5, S1, S2).
7. Tests: component tests, robustness tests, CI (T1–T4).
8. README restructure, LICENSE, pipeline doc, screenshots.

## Resolution

| # | Status | What was done |
|---|---|---|
| F1 | Done | Active sessions older than `STALE_SESSION_HOURS` are marked `abandoned` when the user starts a new session (test: `test_stale_active_sessions_are_abandoned`). |
| F2 | Done | `pose/confidence.ts`: per-frame confidence (visibility × jitter) and per-rep confidence (mean × dropout × frame rate). Reps below 0.5 are counted but not scored. |
| F3 | Done | Frames below 0.4 confidence give no live cues and no red joints. |
| F4 | Done | "Can't see your *knees* clearly" / "Rep counted, form not scored: …" with the weakest body part named. |
| F5 | Done | Styled confirm dialog, toasts, skeleton loaders (`components/Feedback.tsx`). |
| F6 | Done | Unused constant removed. |
| D1 | Kept, documented | The Python rule mirror is still needed for the baseline and the seed script. A pytest keeps its thresholds in sync. Its hip-pike blind spot (pike scored as sag, 10 points harsher) is documented in `ml/rules.py`. |
| S1 | Done | "synthetic-trained" labels on every model score (session detail, history, charts, progress, About). `/api/model` returns `synthetic: true` and a model card. |
| S2 | Done | Seeded sessions carry `demo: true`, shown as a "demo data" tag and a note on Progress. |
| S3 | Done | README separates synthetic results from real-clip results. |
| M1 | Done, small | Manifest + fetch + landmark extraction + replay through the TS engine (`experiments/real_clips`, `frontend/scripts/eval-clips.ts`). Only the bundled clip could be evaluated: Wikimedia returned HTTP 429 to every API and file request from the build machine for over an hour, so the other candidate clips are listed in the manifest but not labelled. |
| M2 | Done | Unweighted fit. ECE, Brier score and a reliability table are reported for every model (GB ECE 0.009–0.028 on synthetic test). |
| M3 | Done | Majority, rules (threshold tuned on validation), logistic regression, random forest and gradient boosting on identical splits. |
| M4 | Done | Confusion matrix, per-class report, ROC-AUC, PR-AUC. Threshold and abstention confidence are chosen on validation only. |
| M5 | Done | `errors.csv` per run plus a written analysis (`experiments/reports/form_classifier_synthetic.md`, README). |
| M6 | Done | Server-side `out_of_distribution` / `uncertain` / `not_scored` model statuses. Browser-side abstention for poor tracking. |
| M7 | Done | Evaluation seed 2024 ≠ tuning seed 3. Aggregation bug in the multi-exercise false-alarm rate fixed and re-run. Tests cover detection and false-alarm rates. |
| M8 | Done | `form_model.card.json` (version, date, commit, split hashes and sizes, synthetic flag, feature version, hyper-parameters, seed, library versions, metrics). |
| M9 | Done | The README "AI/ML Pipeline" table maps stage → file → parameters. The About page shows the same steps. |
| X1 | Done | Placeholder or short `JWT_SECRET`: refuses to start unless `ENV=development`, where an ephemeral secret is used with a warning. `.env.example` explains how to generate one. |
| X2 | Done | Failed-login limiter per email + IP (429). |
| X3 | Done | ASGI body-limit middleware counts streamed bytes (test with a chunked body). |
| X4 | Done | Log format and level configured. Access log records method, path, status and duration only. |
| X5 | Done | `/api/health` returns `status: degraded` when the model is missing or corrupt, and 503 when MongoDB is unreachable. |
| X6, X7 | No change needed | — |
| U1 | Done | Disclaimer in the app footer, on the About page and in the README. |
| U2 | Done | Confidence meter in the HUD and the session detail. Per-rep confidence and abstention reasons in the rep table. |
| U3 | Done | About page: pipeline, data sent, model card, baseline table, real-video results, limits. |
| U4, U5 | Done | See F5. |
| T1 | Done | vitest + Testing Library component tests (HUD abstention, confidence meter, confirm dialog, login validation, About synthetic label, degraded model, real-eval panel). |
| T2 | Done | `tests/test_robustness.py` (empty session, malformed payloads, chunked oversize body, NaN features, low-confidence reps, old clients, OOD, uncertain, missing and corrupt model, API without a model, rate limit, secret policy, DB down). |
| T3 | Done | Fatigue false-alarm and detection-rate tests. |
| T4 | Done, not run here | `.github/workflows/ci.yml` (pytest with a MongoDB service, vitest, build). |
