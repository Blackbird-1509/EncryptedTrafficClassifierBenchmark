"""Feature extraction from (sizes, ts) arrays.

seq_features : (B, 3, N) tensor for the neural nets -> [signed size / 1500, log(1 + IAT ms) / 8, direction]
flow_stats   : (B, F) matrix of classic flow statistics for XGBoost, computed over the SAME first-N window
               the neural nets see, so the comparison is fair.
No IP addresses or ports are used anywhere.
"""
import numpy as np

N_MODEL = 32  # observation window (packets)


def _iat(ts, valid):
    iat = np.diff(ts, axis=1, prepend=ts[:, :1])
    return np.where(valid, np.maximum(iat, 0), 0)


def seq_features(sizes, ts, n=N_MODEL):
    s, t = sizes[:, :n], ts[:, :n]
    valid = s != 0
    iat_ms = _iat(t, valid) * 1000.0
    return np.stack([s / 1500.0, np.log1p(iat_ms) / 8.0, np.sign(s)], axis=1).astype(np.float32)


def _mstats(v, m):
    """mean / std / min / max of v over mask m (rows with no entries give zeros)."""
    cnt = m.sum(1)
    safe = np.maximum(cnt, 1)
    mean = (v * m).sum(1) / safe
    std = np.sqrt((((v - mean[:, None]) ** 2) * m).sum(1) / safe)
    mn = np.where(cnt > 0, np.where(m, v, np.inf).min(1), 0)
    mx = np.where(cnt > 0, np.where(m, v, -np.inf).max(1), 0)
    return [mean, std, mn, mx]


def flow_stats(sizes, ts, n=N_MODEL):
    s = sizes[:, :n].astype(np.float64)
    t = ts[:, :n].astype(np.float64)
    valid = s != 0
    size = np.abs(s)
    fwd, bwd = valid & (s > 0), valid & (s < 0)
    iat = _iat(t, valid)
    iat_m = valid & (np.arange(s.shape[1])[None, :] > 0)

    n_all, n_f, n_b = valid.sum(1), fwd.sum(1), bwd.sum(1)
    b_f, b_b = (size * fwd).sum(1), (size * bwd).sum(1)
    dur = np.where(valid, t, 0).max(1)
    cols = [n_all, n_f, n_b, b_f, b_b, dur,
            n_all / np.maximum(dur, 1e-3), (b_f + b_b) / np.maximum(dur, 1e-3),
            n_f / np.maximum(n_all, 1), b_f / np.maximum(b_f + b_b, 1),
            size[:, 0], iat[:, 1] if s.shape[1] > 1 else np.zeros(len(s))]
    for m in (valid, fwd, bwd):
        cols += _mstats(size, m)
    cols += _mstats(iat, iat_m)
    # packet-length histogram (fractions)
    edges = [0, 100, 200, 400, 800, 1200, 1400, 1501]
    for lo, hi in zip(edges[:-1], edges[1:]):
        cols.append(((size > lo) & (size <= hi) & valid).sum(1) / np.maximum(n_all, 1))
    return np.stack(cols, axis=1).astype(np.float32)
