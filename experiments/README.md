# Experiments

Each experiment is a JSON config in `configs/`. Run it with:

```bash
python -m experiments.run --config experiments/configs/<name>.json
```

A run writes `results/<timestamp>-<name>/`:

- `config.json`: a snapshot of the config
- `env.json`: timestamp, git commit (with `-dirty` if the tree had changes), Python / NumPy / scikit-learn / joblib versions, platform
- `metrics.json`: all metrics
- task-specific files: `errors.csv` (classifier: most confident misses and false alarms),
  `model_card.json`, `per_clip.json` and `errors.json` (real clips)

Only runs that were actually executed are committed. Write-ups comparing runs go in `reports/`.

## Adding an experiment

1. Copy a config and change `name`, `description` and the parameters.
2. For a new task type, add a branch in `run.py` that returns a metrics dict.
3. Run it, commit the results folder, and add a row below.

## Runs

| Run | Data | Headline | Report |
|---|---|---|---|
| `20260916-144949-form_classifier_synthetic` | **synthetic**, 6,000 / 2,000 / 2,000 reps per exercise | ROC-AUC 0.95–1.00 (model) vs 0.87–0.97 (rules); ECE 0.009–0.028 | [form_classifier_synthetic.md](reports/form_classifier_synthetic.md) |
| `20260916-145023-form_classifier_small_data` | **synthetic**, 500 training reps | ROC-AUC 0.001–0.034 lower than with 6,000 reps | same |
| `20260916-145038-fatigue_synthetic` | **synthetic** 12-rep sets, 300 per pattern | same rates as below; `false_alarm_rate` stored as null (aggregation bug, fixed) | [fatigue_synthetic.md](reports/fatigue_synthetic.md) |
| `20260916-154342-fatigue_synthetic` | **synthetic**, as above | false alarms 4.0%; detection 24% (mild) / 56% (strong) / 77% (sudden) | same |
| `20260916-162813-browser_parity` | real, bundled sample clip | Browser WASM vs Python landmarks: mean diff 0.004, p95 0.011 frame units; 4/4 reps in both | [real_clips_repcount.md](reports/real_clips_repcount.md) |
| `20260916-163224-real_clips_repcount` | **real** video, 11 clips / 217 reps (6 filmed, 5 rendered avatars) | Count MAE 4.0 reps/clip, exact 64%, within ±1 82%; hand-verified subset MAE 0.2 | [real_clips_repcount.md](reports/real_clips_repcount.md) |
| `20261126-170343-cv_classification` | **synthetic** 3-D skeleton sequences (2,000 train / 800 test / 600 hard / 600 sweep) | Exercise macro F1: frame 0.987 / stats 0.995 / TCN 0.993; unseen views 0.82 / 0.94 / 0.94; hard 0.967 / 0.918 / 0.972 | [docs/COMPUTER_VISION.md](../docs/COMPUTER_VISION.md) §5 |
| `20261126-171003-cv_repcount` | **synthetic**, through the TS engine | Rep MAE 1.74, precision 1.000, recall 0.785; false reps with the wrong exercise selected 4.2/min → 0.018/min with the TCN gate; posture P/R 0.77/0.63; side views 2–3× better than front/rear | §6, §8, §9 |
| `20261126-145152-cv_landmarks` | **real** frames, bundled clip; reference = MediaPipe heavy | lite PCK@0.1 0.887 at 13 ms, full 0.945 at 18 ms, heavy 57 ms; low light: enhancement hurts; boxed joints keep visibility 0.71 at PCK@0.1 0.15 | §6, §7 |
| `20261126-coco_pck` | **real** images, 493 single-person COCO val photos, human labels | lite PCK@0.2 0.760 (0.822 detected only), OKS 0.789; full 0.796 / 0.813; heavy 0.819 / 0.832 | §7 |
| `20261126-165409-real_clips_repcount` | **real** video, 11 clips / 217 reps (re-run, MediaPipe 1.2.3) | Count MAE 4.27, exact 45%, within ±1 73% | [real_clips_repcount.md](reports/real_clips_repcount.md) |
| `20261126-165737-cv_real_clips` | **real** video, BEFORE the arm-posture fix | TCN called front-view squats "lunge" in 100% of windows; window accuracy 0.57; the gate blocked real reps. Kept as evidence of the shortcut | §7 |
| `20261126-171149-cv_real_clips` | **real** video, 11 clips | False reps with the wrong exercise selected 9.42 → 0.15/min with the gate; gate vetoed 26 real reps (24 kneeling-curl variant); exercise window accuracy frame 0.83 / TCN 0.78 | §5, §8 |
| posture precision/recall on real video | — | Dataset required before evaluation (no coach-labelled real reps). | — |
| real-video form accuracy | — | Dataset required before evaluation. (No coach-labelled real reps.) | — |
| fatigue on real sets | — | Dataset required before evaluation. | — |
