"""Exercise classification from pose: a per-frame (pose-based) model vs temporal models.

Three models answer the same question - "which exercise is this 2-second window?" (5 exercises +
"other" for non-exercise movement) - from the same normalised landmarks:

  frame     Pose-based. A gradient-boosted tree classifier sees ONE frame's features (normalised
            joint coordinates, joint angles, visibility). A window's prediction is the mean of its
            frame probabilities, so the model never sees how the pose changes over time.
  stats     Hand-crafted temporal features: mean / std / min / max / mean |velocity| of the angle
            and position features over the window, fed to gradient-boosted trees.
  tcn       Learned temporal model: a 1-D temporal convolutional network (3 dilated conv layers,
            receptive field 23 frames = 1.5 s, global average + max pooling) over the window's
            frame features. Trained with JAX, served with a ~40-line NumPy forward pass.

Windows are 30 frames (2 s at 15 fps). Exercise windows are only used when at least half of their
frames are inside the set (not the lead-in standing still).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .dataset import DATA, load_meta
from .normalize import FRAME_FEATURES, frame_features, frames_to_array, mirror
from .skeleton import CLASSES

WIN = 30
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
ANGLE_FEATS = [i for i, f in enumerate(FRAME_FEATURES) if not f.startswith(("x", "v"))]  # angles, y, trunk, ...


@dataclass
class Seq:
    id: str
    label: int
    view: str
    feats: np.ndarray   # [T, F]
    active: np.ndarray  # [T] bool
    meta: dict


def load_split(name: str, augment_mirror=False, limit: int | None = None) -> list[Seq]:
    out = []
    for m in load_meta(name)[:limit]:
        s = json.loads((DATA / name / f"{m['id']}.json").read_text())
        lm, present = frames_to_array(s["frames"])
        aspect = s["width"] / s["height"]
        act = np.array(s["active"], bool)
        out.append(Seq(m["id"], CLS_IDX[m["label"]], m["camera"]["view"], frame_features(lm, present, aspect), act, m))
        if augment_mirror:
            lm2 = lm.copy()
            ok = ~np.isnan(lm2[:, 0, 0])
            lm2[ok] = mirror(lm2[ok])
            out.append(Seq(m["id"] + "~m", CLS_IDX[m["label"]], m["camera"]["view"], frame_features(lm2, present, aspect), act, m))
    return out


def windows(seqs: list[Seq], stride: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    X, y, where = [], [], []
    for si, s in enumerate(seqs):
        T = len(s.feats)
        for st in range(0, max(1, T - WIN + 1), stride):
            if T < WIN:
                break
            if s.label != CLS_IDX["other"] and s.active[st:st + WIN].mean() < 0.5:
                continue
            X.append(s.feats[st:st + WIN]); y.append(s.label); where.append((si, st))
    return np.array(X, np.float32), np.array(y), where


def stats_features(W: np.ndarray) -> np.ndarray:
    A = W[:, :, ANGLE_FEATS]
    vel = np.abs(np.diff(A, axis=1)).mean(1)
    return np.concatenate([A.mean(1), A.std(1), A.min(1), A.max(1), vel], -1)


# ------------------------------------------------------------------------------------------------
# temporal convolutional network: JAX for training, NumPy for inference
# ------------------------------------------------------------------------------------------------

TCN_LAYERS = [(5, 1), (5, 2), (3, 4)]  # (kernel, dilation)
HID = 48


def tcn_init(F: int, n_cls: int, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    p, cin = {}, F
    for i, (k, _) in enumerate(TCN_LAYERS):
        p[f"w{i}"] = (rng.normal(0, math.sqrt(2 / (k * cin)), (k, cin, HID))).astype(np.float32)
        p[f"b{i}"] = np.zeros(HID, np.float32)
        cin = HID
    p["wo"] = (rng.normal(0, math.sqrt(1 / (2 * HID)), (2 * HID, n_cls))).astype(np.float32)
    p["bo"] = np.zeros(n_cls, np.float32)
    return p


def tcn_forward_np(p: dict, x: np.ndarray) -> np.ndarray:
    """x [B, T, F] (already standardised) -> logits [B, C]. 'same' padding, ReLU."""
    h = x
    for i, (k, d) in enumerate(TCN_LAYERS):
        w, b = p[f"w{i}"], p[f"b{i}"]
        pad = d * (k - 1) // 2
        hp = np.pad(h, ((0, 0), (pad, pad), (0, 0)))
        T = h.shape[1]
        out = np.zeros((h.shape[0], T, w.shape[2]), np.float32) + b
        for j in range(k):
            out += hp[:, j * d:j * d + T, :] @ w[j]
        h = np.maximum(out, 0)
    z = np.concatenate([h.mean(1), h.max(1)], -1)
    return z @ p["wo"] + p["bo"]


def _jax_forward(p, x):
    import jax
    import jax.numpy as jnp
    h = x
    for i, (k, d) in enumerate(TCN_LAYERS):
        pad = d * (k - 1) // 2
        h = jax.lax.conv_general_dilated(h, p[f"w{i}"], (1,), [(pad, pad)], rhs_dilation=(d,),
                                         dimension_numbers=("NWC", "WIO", "NWC")) + p[f"b{i}"]
        h = jax.nn.relu(h)
    z = jnp.concatenate([h.mean(1), h.max(1)], -1)
    return z @ p["wo"] + p["bo"]


def _sample_batch(seqs: list[Seq], rng, n: int, mu, sd):
    """On-the-fly windows with time-warp (speed 0.8-1.25x), feature noise and joint dropout."""
    xs, ys = [], []
    while len(xs) < n:
        s = seqs[int(rng.integers(len(seqs)))]
        L = int(rng.integers(24, 38))
        T = len(s.feats)
        if T < L + 1:
            continue
        st = int(rng.integers(0, T - L))
        if s.label != CLS_IDX["other"] and s.active[st:st + L].mean() < 0.5:
            continue
        src = s.feats[st:st + L]
        pos = np.linspace(0, L - 1, WIN)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, L - 1)
        a = (pos - lo)[:, None]
        w = src[lo] * (1 - a) + src[hi] * a
        w = w + rng.normal(0, 0.02, w.shape).astype(np.float32)
        if rng.random() < 0.3:  # drop a joint group for the whole window (occlusion)
            j = int(rng.integers(12))
            w[:, [j, 12 + j, 24 + j]] = 0.0
        xs.append(w); ys.append(s.label)
    X = (np.array(xs, np.float32) - mu) / sd
    return X, np.array(ys)


def train_tcn(train: list[Seq], val_X, val_y, steps=2500, batch=256, lr=2e-3, seed=0, log=print):
    import jax
    import jax.numpy as jnp

    F = train[0].feats.shape[1]
    allf = np.concatenate([s.feats for s in train[::3]])
    mu, sd = allf.mean(0), allf.std(0) + 1e-3
    p = {k: jnp.asarray(v) for k, v in tcn_init(F, len(CLASSES), seed).items()}
    m = {k: jnp.zeros_like(v) for k, v in p.items()}
    v = {k: jnp.zeros_like(v) for k, v in p.items()}
    wd = 1e-4

    def loss_fn(p, x, y):
        logits = _jax_forward(p, x)
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), y[:, None], 1))

    @jax.jit
    def step(p, m, v, x, y, i, lr_t):
        loss, g = jax.value_and_grad(loss_fn)(p, x, y)
        b1, b2 = 0.9, 0.999
        m = {k: b1 * m[k] + (1 - b1) * g[k] for k in p}
        v = {k: b2 * v[k] + (1 - b2) * g[k] ** 2 for k in p}
        p = {k: p[k] - lr_t * ((m[k] / (1 - b1 ** i)) / (jnp.sqrt(v[k] / (1 - b2 ** i)) + 1e-8) + wd * p[k]) for k in p}
        return p, m, v, loss

    rng = np.random.default_rng(seed)
    vx = (val_X - mu) / sd
    best, best_p, hist = -1.0, None, []
    t0 = time.time()
    for i in range(1, steps + 1):
        x, y = _sample_batch(train, rng, batch, mu, sd)
        lr_t = lr * 0.5 * (1 + math.cos(math.pi * i / steps))
        p, m, v, loss = step(p, m, v, jnp.asarray(x), jnp.asarray(y), i, lr_t)
        if i % 250 == 0 or i == steps:
            pn = {k: np.asarray(a) for k, a in p.items()}
            f1 = f1_score(val_y, tcn_forward_np(pn, vx).argmax(1), average="macro")
            hist.append({"step": i, "loss": round(float(loss), 4), "val_macro_f1": round(float(f1), 4)})
            log(f"  tcn step {i} loss {float(loss):.3f} val F1 {f1:.3f} ({time.time() - t0:.0f}s)")
            if f1 > best:
                best, best_p = f1, pn
    return {"params": best_p, "mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "history": hist,
            "train_s": round(time.time() - t0, 1)}


def tcn_predict(model: dict, X: np.ndarray) -> np.ndarray:
    z = tcn_forward_np(model["params"], (X - model["mu"]) / model["sd"])
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def save_tcn(model: dict, path: Path) -> None:
    np.savez_compressed(path, mu=model["mu"], sd=model["sd"], **{f"p_{k}": v for k, v in model["params"].items()})


def load_tcn(path: Path) -> dict:
    z = np.load(path)
    return {"params": {k[2:]: z[k] for k in z.files if k.startswith("p_")}, "mu": z["mu"], "sd": z["sd"]}


# ------------------------------------------------------------------------------------------------
# evaluation helpers
# ------------------------------------------------------------------------------------------------

def report(y, pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)), zero_division=0)
    return {
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "accuracy": round(float((y == pred).mean()), 4),
        "per_class": {c: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                          "f1": round(float(f[i]), 3), "support": int(s[i])} for i, c in enumerate(CLASSES)},
        "confusion": {"labels": CLASSES, "matrix": confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist()},
    }


def fit_frame_model(train: list[Seq], seed=0):
    X, y = [], []
    for s in train:
        idx = np.arange(0, len(s.feats), 3)
        if s.label != CLS_IDX["other"]:
            idx = idx[s.active[idx]]
        X.append(s.feats[idx]); y.append(np.full(len(idx), s.label))
    X, y = np.concatenate(X), np.concatenate(y)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed)
    clf.fit(X, y)
    return clf


def frame_model_predict(clf, W: np.ndarray) -> np.ndarray:
    B, T, F = W.shape
    return clf.predict_proba(W.reshape(-1, F)).reshape(B, T, -1).mean(1)


def fit_stats_model(train_W, train_y, seed=0):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed)
    clf.fit(stats_features(train_W), train_y)
    return clf


def latency_ms(fn, x, repeats=200) -> float:
    fn(x)
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn(x)
        ts.append(time.perf_counter() - t)
    return round(float(np.median(ts)) * 1000, 3)
