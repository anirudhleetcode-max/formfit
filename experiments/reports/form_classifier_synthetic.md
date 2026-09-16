# Form classifier on synthetic reps (runs of 2026-09-16)

Runs compared:

- `results/20260916-144949-form_classifier_synthetic` (6,000 training reps per exercise; this run produced the served model)
- `results/20260916-145023-form_classifier_small_data` (500 training reps per exercise)

Both runs use the same seed and the same independently generated validation and test sets
(2,000 reps each, identical hashes in `metrics.json`). **All data is synthetic** and comes from
`backend/ml/simulate.py`. Nothing here says how the model behaves on real lifters.

## Baselines vs model (test set, 6,000-rep run)

| Exercise | Majority acc. | Rules acc. / ROC-AUC | Log. reg. acc. / ROC-AUC | Selected model | Model acc. / ROC-AUC | Model ECE / Brier | Rules ECE |
|---|---|---|---|---|---|---|---|
| Squat | 0.730 | 0.834 / 0.866 | 0.797 / 0.868 | gradient boosting | 0.900 / 0.951 | 0.028 / 0.077 | 0.544 |
| Push-up | 0.494 | 0.967 / 0.968 | 0.892 / 0.965 | gradient boosting | 0.974 / 0.998 | 0.009 / 0.017 | 0.341 |
| Curl | 0.668 | 0.907 / 0.928 | 0.834 / 0.916 | gradient boosting | 0.932 / 0.978 | 0.017 / 0.049 | 0.482 |
| Press | 0.566 | 0.914 / 0.901 | 0.884 / 0.945 | gradient boosting | 0.918 / 0.973 | 0.013 / 0.058 | 0.329 |
| Lunge | 0.482 | 0.863 / 0.870 | 0.798 / 0.886 | random forest | 0.893 / 0.951 | 0.017 / 0.081 | 0.399 |

Notes:

- The model is chosen by validation ROC-AUC. Decision thresholds and the abstention confidence are
  also picked on validation. The test set is used once.
- The rule "probability" is just `score / 100`, so its ECE is large by construction. Rules are a
  ranking baseline, not a calibrated one.
- On push-ups the fixed rules are almost as accurate as the model (0.967 vs 0.974), because the
  simulator's push-up faults are close to single-feature thresholds.
- Logistic regression is well calibrated but has a lower ROC-AUC on squats and lunges, where faults
  interact (depth × lean, valgus only measurable in some views).

## Data volume (500 vs 6,000 training reps)

With 500 reps the selected models lose 0.001–0.034 ROC-AUC (largest on squat: 0.951 → 0.917).
The random forest wins validation on four of five exercises at the small size, and gradient
boosting only catches up with more data. If real labelled reps ever exist, expect only a few
hundred per exercise at first, so the random forest is the safer starting point.

## Error analysis (`errors.csv`: the 15 most confident misses and false alarms per exercise)

Patterns seen in the file:

1. **Missed faults are mostly view-dependent faults.** All 9 missed squat `knee_valgus` reps
   have `valgus_min = 1.0`, the neutral value used when the camera is side-on and valgus cannot be
   measured. The same holds for 12 of the 14 missed lunge valgus reps. These errors are not model
   errors: the information is not in the features. The fix is a camera-setup hint (film squats
   from the front to check knees), not a bigger model.
2. **Press `back_arch` is the hardest fault** (13 of 15 press misses; fault recall 0.78). The
   torso-lean measurement of a standing lifter is small and noisy, so an arch and a clean rep look
   alike.
3. **Curl misses are `elbow_drift` (9) and `torso_swing` (5)**, both measured close to their
   thresholds, so they are borderline reps with measurement noise.
4. **False alarms are clean reps with borderline measurements.** Examples: a squat with a 0.47 s
   descent (the rushed-descent rule is < 0.5 s), a squat with 47° measured lean whose true lean is
   under the 45° limit, and presses whose measured lockout (160–173°) is near the 158° limit. The
   label is computed on the true parameters, while the model only sees noisy ones, so some of this
   error cannot be removed.

## What this does not show

- Real-lifter accuracy: **not measured yet** (no labelled real reps).
- Whether the simulator's fault mixture resembles real gym populations: unknown.
