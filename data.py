"""Data ingestion for encrypted traffic-type classification.

Every flow is stored as two arrays of length N_STORE (=64):
    sizes : signed packet size in bytes (+ = initiator -> responder, - = reverse, 0 = padding)
    ts    : packet timestamp in seconds, relative to the first packet of the flow

The model only looks at the first 32 packets, but we keep 64 so that packet-loss
augmentation still leaves enough packets to fill the model window.
"""
import os
from collections import OrderedDict

import numpy as np

N_STORE = 64
CLASSES = ["browsing", "chat", "email", "file_transfer", "p2p", "streaming", "voip"]

# ISCXVPN2016 filename -> traffic category. First matching rule wins (order matters).
# CHECK THIS against your download: run prepare_data.py and look at the printed counts.
LABEL_RULES = [
    ("streaming", ["youtube", "vimeo", "netflix", "spotify"]),
    ("p2p", ["torrent"]),
    ("chat", ["chat"]),                      # facebook_chat, hangouts_chat, gmailchat, skype_chat ...
    ("email", ["email", "mail"]),
    ("file_transfer", ["sftp", "ftps", "scp", "file"]),
    ("voip", ["voip", "audio", "video"]),    # voipbuster, skype_audio, facebook_video ...
    ("browsing", ["firefox", "chrome", "browsing"]),
]
IGNORE_PORTS = {53, 5353, 5355, 67, 68, 137, 138, 1900}   # DNS/mDNS/LLMNR/DHCP/NetBIOS/SSDP


def label_from_filename(path):
    name = os.path.basename(path).lower()
    for cls, keys in LABEL_RULES:
        if any(k in name for k in keys):
            return cls
    return None


# --------------------------------------------------------------------------- pcap parsing
def pcap_to_flows(path, max_pkts=N_STORE, timeout=120.0, min_pkts=5, max_flows=None, seed=0):
    """Parse one pcap/pcapng into bidirectional flows. Returns (sizes, ts) float32 arrays."""
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
    from scapy.utils import PcapReader

    flows, finished = {}, []

    def finish(f):
        if len(f["sizes"]) >= min_pkts:
            finished.append(f)

    with PcapReader(path) as reader:
        for pkt in reader:
            if IP in pkt:
                ip = pkt[IP]
                src, dst, proto, size = ip.src, ip.dst, ip.proto, int(ip.len)
            elif IPv6 in pkt:
                ip = pkt[IPv6]
                src, dst, proto, size = ip.src, ip.dst, ip.nh, int(ip.plen) + 40
            else:
                continue
            if TCP in pkt:
                sp, dp = pkt[TCP].sport, pkt[TCP].dport
            elif UDP in pkt:
                sp, dp = pkt[UDP].sport, pkt[UDP].dport
            else:
                continue
            if sp in IGNORE_PORTS or dp in IGNORE_PORTS:
                continue
            t = float(pkt.time)
            a, b = (src, sp), (dst, dp)
            key = (proto,) + ((a, b) if a <= b else (b, a))
            f = flows.get(key)
            if f is not None and t - f["last"] > timeout:      # idle timeout -> new flow
                finish(f)
                f = None
            if f is None:
                f = flows[key] = {"init": a, "t0": t, "last": t, "sizes": [], "ts": []}
            f["last"] = t
            if len(f["sizes"]) < max_pkts:
                sign = 1 if a == f["init"] else -1
                f["sizes"].append(sign * max(1, min(size, 1500)))
                f["ts"].append(t - f["t0"])
    for f in flows.values():
        finish(f)

    if max_flows is not None and len(finished) > max_flows:
        idx = np.random.default_rng(seed).choice(len(finished), max_flows, replace=False)
        finished = [finished[i] for i in idx]
    sizes = np.zeros((len(finished), max_pkts), np.float32)
    ts = np.zeros((len(finished), max_pkts), np.float32)
    for i, f in enumerate(finished):
        n = len(f["sizes"])
        sizes[i, :n] = f["sizes"]
        ts[i, :n] = f["ts"]
    return sizes, ts


def _worker(args):
    path, max_flows = args
    try:
        return path, pcap_to_flows(path, max_flows=max_flows)
    except Exception as e:  # corrupt / truncated capture
        print(f"  ! skipped {os.path.basename(path)}: {e}")
        return path, None


def build_from_pcap_dir(root, workers=1, max_flows_per_file=None):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs
                   if f.lower().endswith((".pcap", ".pcapng")))
    todo, unlabeled = [], []
    for p in files:
        (todo if label_from_filename(p) else unlabeled).append(p)
    print(f"{len(files)} captures found, {len(todo)} labelled, {len(unlabeled)} without a matching rule")
    for p in unlabeled:
        print("   no label rule for:", os.path.basename(p))

    jobs = [(p, max_flows_per_file) for p in todo]
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:  # same behaviour on Windows/Linux/macOS
            results = list(ex.map(_worker, jobs))
    else:
        results = [_worker(j) for j in jobs]

    S, T, Y, G = [], [], [], []
    for gid, (p, res) in enumerate(results):
        if res is None or len(res[0]) == 0:
            continue
        s, t = res
        S.append(s); T.append(t)
        Y.append(np.full(len(s), CLASSES.index(label_from_filename(p))))
        G.append(np.full(len(s), gid))
        print(f"  {os.path.basename(p):45s} {len(s):7d} flows -> {label_from_filename(p)}")
    return np.concatenate(S), np.concatenate(T), np.concatenate(Y), np.concatenate(G)


# --------------------------------------------------------------------------- synthetic data
# Rough per-class behaviour (size mix, direction persistence, inter-arrival time).
# Purely for smoke-testing the pipeline without a dataset -- NOT a substitute for real traffic.
_PROFILES = {
    #               (p_small, p_mid, p_large), small_mu, small_sd, p_stay_dir, p_down, iat_mu (ln s), iat_sigma
    "browsing":      ((.35, .25, .40), 90, 40, .65, .65, -4.6, 1.7),
    "chat":          ((.75, .22, .03), 110, 45, .35, .50, -1.2, 1.0),
    "email":         ((.30, .50, .20), 100, 40, .60, .55, -3.4, 1.3),
    "file_transfer": ((.30, .02, .68), 60, 10, .85, .50, -7.0, 1.0),
    "p2p":           ((.35, .40, .25), 120, 60, .50, .50, -4.0, 2.0),
    "streaming":     ((.25, .10, .65), 70, 20, .80, .85, -5.0, 1.4),
    "voip":          ((.97, .02, .01), 130, 30, .30, .50, -3.9, 0.35),
}


def make_synthetic(n_per_class=1500, n_files_per_class=12, seed=0):
    rng = np.random.default_rng(seed)
    S, T, Y, G = [], [], [], []
    gid = 0
    for ci, cls in enumerate(CLASSES):
        w0, mu_s, sd_s, stay0, down0, iat_mu0, iat_sd0 = _PROFILES[cls]
        for _ in range(n_per_class):
            file_id = gid + rng.integers(n_files_per_class)
            # per-flow jitter of the class parameters -> overlapping classes
            w = np.array(w0) * rng.lognormal(0, 0.5, 3); w /= w.sum()
            stay = np.clip(stay0 + rng.normal(0, .12), .05, .95)
            down = np.clip(down0 + rng.normal(0, .12), .05, .95)
            iat_mu = iat_mu0 + rng.normal(0, .6)
            n = rng.integers(8, N_STORE + 1)
            comp = rng.choice(3, n, p=w)
            small = np.clip(rng.normal(mu_s, sd_s, n), 40, 220)
            mid = rng.uniform(220, 1200, n)
            large = np.clip(rng.normal(1430, 40, n), 1000, 1500)
            size = np.where(comp == 0, small, np.where(comp == 1, mid, large))
            d = np.empty(n)
            d[0] = 1
            cur = 1
            for i in range(1, n):
                if rng.random() > stay:                    # switch direction
                    cur = -1 if rng.random() < down else 1
                d[i] = cur
            iat = rng.lognormal(iat_mu, iat_sd0, n)
            iat[0] = 0
            sizes = np.zeros(N_STORE, np.float32); ts = np.zeros(N_STORE, np.float32)
            sizes[:n] = (d * np.round(size)).astype(np.float32)
            ts[:n] = np.cumsum(iat)
            S.append(sizes); T.append(ts); Y.append(ci); G.append(file_id)
        gid += n_files_per_class
    return np.stack(S), np.stack(T), np.array(Y), np.array(G)


# --------------------------------------------------------------------------- save / load / clean / split
def save(path, sizes, ts, y, groups):
    np.savez_compressed(path, sizes=sizes, ts=ts, y=y, groups=groups, classes=np.array(CLASSES))


def load(path, min_per_class=30, dedup=True):
    z = np.load(path, allow_pickle=False)
    sizes, ts, y, groups, classes = z["sizes"], z["ts"], z["y"], z["groups"], [str(c) for c in z["classes"]]

    if dedup:  # identical flows split across train/test are a classic leakage source
        seen, keep = set(), []
        for i in range(len(y)):
            key = sizes[i].tobytes() + np.round(ts[i] * 1000).astype(np.int32).tobytes()
            if key not in seen:
                seen.add(key); keep.append(i)
        if len(keep) < len(y):
            print(f"dedup: removed {len(y) - len(keep)} duplicate flows")
        keep = np.array(keep)
        sizes, ts, y, groups = sizes[keep], ts[keep], y[keep], groups[keep]

    counts = np.bincount(y, minlength=len(classes))      # drop tiny classes, re-index labels
    kept = [c for c in range(len(classes)) if counts[c] >= min_per_class]
    remap = {c: i for i, c in enumerate(kept)}
    m = np.isin(y, kept)
    sizes, ts, groups = sizes[m], ts[m], groups[m]
    y = np.array([remap[c] for c in y[m]])
    return sizes, ts, y, groups, [classes[c] for c in kept]


def make_splits(y, groups, mode="flow", seed=42):
    """70/15/15 split. mode='flow': stratified by flow. mode='group': whole pcap files stay together
    (stricter: flows from one capture session never appear in both train and test)."""
    idx = np.arange(len(y))
    if mode == "group":
        from sklearn.model_selection import StratifiedGroupKFold
        folds = [te for _, te in StratifiedGroupKFold(7, shuffle=True, random_state=seed).split(idx, y, groups)]
        te, va = folds[0], folds[1]
        tr = np.concatenate(folds[2:])
        return tr, va, te
    from sklearn.model_selection import train_test_split
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    return tr, va, te
