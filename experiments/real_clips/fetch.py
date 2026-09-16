"""Download the clips listed in data/real_clips/manifest.json into data/real_clips/cache/ (git-ignored).

    python -m experiments.real_clips.fetch            # all clips
    python -m experiments.real_clips.fetch squat_kb_goblet

Files are checked against the sha256 in the manifest. Wikimedia asks for a descriptive
User-Agent and rate-limits aggressively, so downloads are sequential with back-off.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "real_clips" / "manifest.json"
CACHE = ROOT / "data" / "real_clips" / "cache"
UA = {"User-Agent": "FormFit-eval/1.0 (student portfolio project; evaluation of rep counting)"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def clip_path(clip: dict) -> Path:
    if clip.get("local_path"):
        return ROOT / clip["local_path"]
    return CACHE / f"{clip['id']}{Path(clip['url']).suffix.lower()}"


def fetch(clip: dict, retries: int = 5) -> Path | None:
    out = clip_path(clip)
    if out.exists() and (not clip.get("sha256") or sha256(out) == clip["sha256"]):
        return out
    if clip.get("local_path"):
        print(f"[fetch] {clip['id']}: local file missing: {out}")
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(clip["url"], headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(out.with_suffix(".part"), "wb") as fh:
                shutil.copyfileobj(r, fh)
            out.with_suffix(".part").replace(out)
            break
        except Exception as e:  # 429 from Wikimedia is common
            wait = 30 * (attempt + 1)
            print(f"[fetch] {clip['id']}: {e}; retrying in {wait}s")
            time.sleep(wait)
    else:
        return None
    digest = sha256(out)
    if clip.get("sha256") and digest != clip["sha256"]:
        print(f"[fetch] {clip['id']}: sha256 mismatch ({digest}), file changed upstream")
        return None
    print(f"[fetch] {clip['id']}: ok ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main(ids: list[str]) -> None:
    for clip in load_manifest()["clips"]:
        if ids and clip["id"] not in ids:
            continue
        fetch(clip)
        time.sleep(3)


if __name__ == "__main__":
    main(sys.argv[1:])
