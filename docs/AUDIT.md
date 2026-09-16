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

Filled in at the end of phase 2 — see the table below.
