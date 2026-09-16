# Fatigue change-point detector on synthetic sets

Runs: `results/20260916-145038-fatigue_synthetic` and `results/20260916-154342-fatigue_synthetic`.
The two runs used the same config and seed and gave identical flag rates. The first run stored
`false_alarm_rate: null` because of an aggregation bug with exercise-prefixed keys. The second run
was made after the fix and adds `false_alarm_rate` and `detection_rate`.

**Data: synthetic.** Each trial is a 12-rep set from `ml/simulate.py`, with a fault rate of 0.1 and a side view.
Fatigue is injected as a per-rep effect (slower lifting, less range of motion, more faults).
There are 300 trials per pattern per exercise, with evaluation seed 2024. The thresholds were chosen
while looking at seed 3.

## Method (`backend/app/services/analytics.py`)

1. Fatigue index per rep = mean of three relative changes against the median of the first 3 reps:
   concentric time increase, ROM decrease and score decrease (/100). Unscored reps (low tracking
   confidence) contribute 0 to the score part.
2. Single change point: the split `k` (at least 3 reps on each side) that minimises the within-segment
   sum of squares, as in a least-squares mean-shift model.
3. Flag the set when the shift in mean is ≥ 0.12, the Welch t-statistic is ≥ 3.0, and the overall
   linear slope of the index is positive. Sets shorter than 6 reps are never flagged.

## Results (flag rate)

| Pattern | Squat | Push-up | Curl | Mean |
|---|---|---|---|---|
| no fatigue (false alarm) | 0.053 | 0.023 | 0.043 | **0.040** |
| mild progressive | 0.267 | 0.240 | 0.213 | 0.240 |
| strong progressive | 0.593 | 0.537 | 0.540 | 0.557 |
| sudden drop after rep 8 | 0.810 | 0.770 | 0.727 | 0.769 |

Onset localisation on the sudden pattern, where the true onset is rep 9: the mean absolute error is
0.25 / 0.37 / 0.34 reps, and the onset is exact in 79% / 69% / 73% of detected sets.

## Reading

- The detector is conservative: it raises about 4 false alarms per 100 fatigue-free sets and misses most
  mild slowdowns. A single mean-shift model suits a sudden drop better than a gradual drift, so
  progressive fatigue is detected less often.
- A t-test on a searched split point is optimistic (a multiple-comparisons problem). The threshold of
  3.0 is set higher than usual to compensate, and the false-alarm rate above was measured, not assumed.
- Real sets have not been evaluated: **dataset required before evaluation** (sets with a known
  fatigue onset, e.g. from velocity-based training logs).
