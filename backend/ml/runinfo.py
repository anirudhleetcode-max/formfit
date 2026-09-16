"""Environment / provenance info recorded with every run and model card."""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def env_info() -> dict:
    import joblib
    import numpy
    import sklearn
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
