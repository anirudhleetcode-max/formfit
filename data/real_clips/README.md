# Real exercise clips (evaluation data)

`manifest.json` lists public videos used to evaluate rep counting on **real** footage. Only the
manifest is committed. The clips are downloaded on demand into `cache/`, and the extracted landmarks go into
`landmarks/`. Both folders are git-ignored because the files are larger than the repo should carry, and CC BY-SA
clips would need their own attribution on redistribution.

```bash
pip install -r backend/requirements-eval.txt           # mediapipe + opencv (Python), evaluation only
python -m experiments.real_clips.fetch                  # download + sha256 check (sequential, polite)
python -m experiments.run --config experiments/configs/real_clips_repcount.json
```

## Manifest format

Each clip entry has these fields:

| Field | Meaning |
|---|---|
| `id` | Slug. Also the landmark file name |
| `exercise` | `squat`, `pushup`, `curl`, `press` or `lunge` |
| `url` / `local_path` | Direct file URL on upload.wikimedia.org, or a repo path for the bundled sample |
| `page` | The Commons file page (licence and author are shown there) |
| `license`, `author` | Attribution. Only licences that allow reuse (CC BY, CC BY-SA, public domain) |
| `sha256` | Hash of the file that was labelled. A mismatch means the file changed upstream and the labels may be wrong |
| `segment` | `[start_s, end_s]` to analyse, or `null` for the whole video |
| `view` | Camera view as a person would describe it |
| `gt_reps` | Completed reps inside the segment |
| `gt_method` | How `gt_reps` was established |
| `fault_labels` | Optional per-rep list of fault codes (`[]` = clean). `null` = not labelled |

## Sources

- `samples/squat_demo.webm`: Wikimedia Commons, CC BY 3.0 (bundled, see `samples/ATTRIBUTION.md`).
- 10 clips from [PushUpBench](https://huggingface.co/datasets/anonymousatom/pushupbench), a
  rep-counting benchmark with per-clip counts. The dataset card declares CC BY 4.0, but the clips
  appear to come from third-party fitness channels, so they are only downloaded, never committed.
  Five show filmed people and five show rendered 3D avatars (`source_type`).
- Wikimedia Commons candidates are listed under `wikimedia_candidates`. They were not downloaded because
  Commons answered HTTP 429 to every request from the build machine.

## How ground truth was established

For the bundled clip and four PushUpBench clips (`gt_verified_by_hand: true`), a contact sheet of frames sampled at 2 fps (ffmpeg `fps=2,tile`, with timestamps) was
inspected by eye. A rep was counted when the lifter went from the top position to the bottom and back.
Partial movements, set-up and walk-outs were not counted. The timestamps of the bottoms are written into
`gt_method` so the count can be re-checked. Every one of these hand counts agreed with the dataset label.
For the other PushUpBench clips the dataset label is used as is, because the movement was too fast
or too small to count on a contact sheet. `gt_method` says so for each clip, and the evaluation
reports a "hand-verified only" subset. Clips where the count was ambiguous (cut edits,
several people, the exercise mixed with other movements) were either given a `segment` or left out.

No clip has `fault_labels`. Judging form reliably needs a qualified coach, so form accuracy on
real video is **not measured yet**.

## Adding a clip

1. Find a video with a reuse licence. Note the file page, author and licence.
2. Add an entry with `gt_reps: null`, run `fetch`, and fill in `sha256`.
3. Count the reps from a contact sheet (or frame by frame) and write down the method.
4. Re-run the experiment. The runner re-extracts landmarks whenever the hash, fps or segment changes.
