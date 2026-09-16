# Rep counting on real video (run 20260916-163224-real_clips_repcount)

**Sample: 11 clips, 217 reps.** One is the bundled Wikimedia Commons squat clip. Ten come from
PushUpBench (Hugging Face, CC BY 4.0 per the dataset card). Five of the PushUpBench clips show
**filmed people** and five show **rendered 3D avatars**. The sample is tiny and not representative:
every number below comes with a wide error bar.

Pipeline under test: MediaPipe PoseLandmarker lite (Python build, 15 analysed fps) →
`frontend/src/pose/engine.ts`, replayed with `frontend/scripts/eval-clips.ts`. The browser-parity run
(`20260916-162813-browser_parity`) found a mean landmark difference of 0.004 frame units between the
WASM and Python builds, and the same rep count on the sample clip.

## Results

| Clip | Source | Exercise / conditions | True | Counted | Form scored | Model status |
|---|---|---|---|---|---|---|
| squat_fitnessscape_back | filmed | back squat, rear-oblique | 4 | 4 | 4 | 1 ok, 3 uncertain |
| pub_squat_129 | rendered | bodyweight squat | 15 | 15 | 15 | 15 ok |
| pub_squat_130 | rendered | bodyweight squat | 12 | 12 | 12 | 12 ok |
| pub_squat_135 | rendered | bodyweight squat | 15 | 15 | 15 | 15 ok |
| pub_lunge_134 | rendered | reverse lunge, alternating | 20 | 19 | 13 | 13 uncertain |
| pub_lunge_008 | rendered | reverse lunge, alternating, 108 s | 50 | 48 | 36 | 36 uncertain |
| pub_curl_085 | filmed | kneeling curl, no weights | 24 | 25 | 19 | 14 ok, 5 uncertain |
| pub_pushup_098 | filmed | knee push-up | 12 | 12 | 12 | 12 ok |
| pub_pushup_088 | filmed | box push-up | 11 | 11 | 11 | 11 out of distribution |
| pub_pushup_046 | filmed | push-up + toe tap, inset video | 14 | 14 | 14 | 14 ok |
| pub_squat_042 | filmed | drop squat (shallow), inset video | 40 | **0** | 0 | — |

| Subset | Clips | MAE (reps/clip) | Exact | Within ±1 | Total counted / true |
|---|---|---|---|---|---|
| All | 11 | 4.00 | 64% | 82% | 175 / 217 |
| Filmed | 6 | 6.83 | 67% | 83% | 66 / 105 |
| Rendered | 5 | 0.60 | 60% | 80% | 109 / 112 |
| Ground truth verified by hand | 5 | 0.20 | 80% | 100% | 65 / 66 |

Excluding the drop-squat clip, the filmed MAE would be 0.2, but that clip is a real failure and it stays in.
86% of counted reps had high enough tracking confidence to be form-scored.

## Error analysis (`errors.json`)

1. **Shallow variant → zero reps.** On the drop-squat clip the knee angle stays above 121° in 95% of
   frames. The squat state machine needs the angle to fall below 115° (the bottom threshold), so no
   rep ever completes. The pipeline is built for full-depth squats; a partial-range variant is
   out of scope. The thresholds were **not** changed after seeing this, because that would tune on the test clip.
2. **The first lunge is missed.** On both lunge clips the first counted rep ends about 3 s after the first
   visible low (7.3 s vs 4 s, 12.2 s vs 9 s). The state machine has to see the lifter at the top before it arms, and the
   first rep starts before that happens.
3. **Movement before the set is counted.** On the curl clip, three "reps" were counted at 1.5–4.1 s,
   while the person was still getting ready. All three were abstained from scoring ("Can't see your
   wrists clearly"), so abstention worked, but the count was still off by one overall.
4. **Low confidence on side-on alternating lunges** (mean frame confidence 0.58–0.59): the far
   ankle is occluded on every other rep, which produces "Can't see your ankles clearly" and 25–32% unscored reps.
5. **The synthetic-trained model is often unsure on real reps.** 83 of 175 reps were `ok`, 57 `uncertain`, 24
   `not_scored` and 11 `out_of_distribution`. All box push-ups are out of range on `torso_sway` / `hip_line_min`,
   because kneeling push-ups bend the shoulder–hip–ankle line far more than the simulator's full push-ups ever do. No fault
   labels exist, so the model's correctness on these reps is **not measured**.

## Caveats

- Five of the eleven ground truths are PushUpBench labels that I could not confirm from contact
  sheets (see `gt_method` in the manifest).
- Rendered avatars have clean, stable landmarks, so they flatter pose tracking. The filmed subset is the
  more relevant one, and it has only 6 clips.
- Wikimedia Commons candidates (kettlebell squats, push-ups, a strict press) are listed in the
  manifest but could not be downloaded (HTTP 429 during the whole session).
