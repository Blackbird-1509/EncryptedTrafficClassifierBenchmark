"""Network-condition perturbations applied directly to packet arrays.

All functions work on batches: sizes (B, L) signed bytes with 0 = padding, ts (B, L) seconds.
Parameters can be scalars (same condition for every flow) or arrays of shape (B,) (per-flow
conditions, used for robustness augmentation during training).

Order of effects mirrors a real path + observer:
    1. packet loss        -> packets vanish, later packets shift forward
    2. delay jitter       -> each packet gets extra delay ~ U(0, jitter); large jitter reorders packets
    3. truncation         -> the observer only sees the first n packets
"""
import numpy as np


def _compact(keep, *arrays):
    """Move kept entries to the front of each row (stable) and zero the rest."""
    order = np.argsort(~keep, axis=1, kind="stable")
    k = np.take_along_axis(keep, order, 1)
    return [np.where(k, np.take_along_axis(a, order, 1), 0) for a in arrays]


def perturb(sizes, ts, loss=0.0, jitter=0.0, trunc=None, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    B, L = sizes.shape
    loss = np.broadcast_to(np.asarray(loss, np.float32), (B,))
    jitter = np.broadcast_to(np.asarray(jitter, np.float32), (B,))
    sizes, ts = sizes.copy(), ts.copy()
    valid = sizes != 0

    if np.any(loss > 0):
        keep = valid & (rng.random((B, L)) >= loss[:, None])
        sizes, ts = _compact(keep, sizes, ts)
        valid = sizes != 0

    if np.any(jitter > 0):
        ts = ts + rng.random((B, L), dtype=np.float32) * jitter[:, None]
        ts = np.where(valid, ts, np.inf)                       # padding sorts to the end
        order = np.argsort(ts, axis=1, kind="stable")
        sizes = np.take_along_axis(sizes, order, 1)
        ts = np.take_along_axis(ts, order, 1)
        valid = sizes != 0
        ts = np.where(valid, ts, 0)

    if trunc is not None:
        n = np.broadcast_to(np.asarray(trunc), (B,))
        cut = np.arange(L)[None, :] >= n[:, None]
        sizes = np.where(cut, 0, sizes)
        ts = np.where(cut, 0, ts)
        valid = sizes != 0

    ts = np.where(valid, ts - ts[:, :1], 0)                    # observer's clock starts at 1st seen packet
    return sizes.astype(np.float32), ts.astype(np.float32)


def sample_conditions(B, rng, p_clean=0.25, max_loss=0.12, max_jitter=0.12, trunc_range=(5, 32), L=64):
    """Random per-flow conditions for robustness training. Returns kwargs for perturb()."""
    loss = np.where(rng.random(B) < 0.7, rng.uniform(0, max_loss, B), 0.0)
    jitter = np.where(rng.random(B) < 0.7, rng.uniform(0, max_jitter, B), 0.0)
    trunc = np.where(rng.random(B) < 0.5, rng.integers(trunc_range[0], trunc_range[1] + 1, B), L)
    clean = rng.random(B) < p_clean
    return dict(loss=np.where(clean, 0, loss), jitter=np.where(clean, 0, jitter),
                trunc=np.where(clean, L, trunc))
