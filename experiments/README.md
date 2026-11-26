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
| `20261126-143915-cv_classification` | **synthetic** 3-D skeleton sequences (2,000 train / 800 test / 600 hard / 600 sweep) | Exercise macro F1: frame 0.993 / stats 0.998 / TCN 0.996; unseen views 0.81 / 0.96 / 0.94; hard 0.981 / 0.940 / 0.985 | [docs/COMPUTER_VISION.md](../docs/COMPUTER_VISION.md) §5 |
| `20261126-144743-cv_repcount` | **synthetic**, through the TS engine | Rep MAE 1.44, precision 1.000, recall 0.823; false reps with the wrong exercise selected 4.5/min → 0.012/min with the TCN gate; posture P/R 0.75/0.64; side views 3× better than front/rear | §6, §8, §9 |
| `20261126-145152-cv_landmarks` | **real** frames, bundled clip; reference = MediaPipe heavy | lite PCK@0.1 0.887 at 13 ms, full 0.945 at 18 ms, heavy 57 ms; low light: enhancement hurts; boxed joints keep visibility 0.71 at PCK@0.1 0.15 | §6, §7 |
| landmark accuracy vs human labels | — | Dataset required before evaluation (`experiments/cv/coco_pck.py`, COCO val2017). | — |
| exercise classification / false reps on real video | — | Dataset required before evaluation (sanity check only: bundled clip 183/183 windows `squat`). | — |
| real-video form accuracy | — | Dataset required before evaluation. (No coach-labelled real reps.) | — |
| fatigue on real sets | — | Dataset required before evaluation. | — |
