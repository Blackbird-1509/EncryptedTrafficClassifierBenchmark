"""Classify the flows of ONE pcap with models trained by run.py.

    python predict.py --pcap test.pcap --results results --model cnn_bigru_attn+aug
    python predict.py --pcap test.pcap --results results --model all           # every trained variant
    python predict.py --pcap test.pcap --results results --model all --loss 0.05 --jitter 0.05 --trunc 10
    python predict.py --pcap test.pcap --results results --model all --csv per_flow.csv

For a fair test the pcap must NOT have been in the folder you ran prepare_data.py on.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

import data as D
from features import flow_stats, seq_features
from models import build
from perturb import perturb


def load_classes(results_dir, data_path):
    p = os.path.join(results_dir, "classes.json")
    if os.path.exists(p):
        return json.load(open(p))
    if data_path:                                  # results made before classes.json existed
        return D.load(data_path)[4]
    raise SystemExit("classes.json not found in the results folder: pass --data flows.npz (the file used for training)")


def find_models(results_dir, which, seed):
    found = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "ckpt", f"*_seed{seed}.*"))):
        variant = os.path.basename(path).rsplit(f"_seed{seed}", 1)[0]
        found[variant] = path
    if which != "all":
        if which not in found:
            raise SystemExit(f"no model '{which}' for seed {seed}. Available: {', '.join(found) or 'none'}")
        found = {which: found[which]}
    if not found:
        raise SystemExit(f"no checkpoints in {results_dir}/ckpt for seed {seed}")
    return found


def predict_proba(variant, path, n_classes, sizes, ts):
    base = variant.partition("+")[0]
    if base == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier()
        clf.load_model(path)
        return clf.predict_proba(flow_stats(sizes, ts))
    model = build(base, n_classes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    X = torch.from_numpy(seq_features(sizes, ts))
    with torch.no_grad():
        return torch.cat([torch.softmax(model(X[i:i + 2048]), 1) for i in range(0, len(X), 2048)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="all", help="e.g. cnn_bigru_attn+aug, xgb, cnn1d ... or 'all'")
    ap.add_argument("--seed", type=int, default=0, help="which training seed's checkpoint to use")
    ap.add_argument("--true-label", help="override the label guessed from the filename")
    ap.add_argument("--data", help="flows.npz used for training (only needed if classes.json is missing)")
    ap.add_argument("--max-flows", type=int, default=5000)
    ap.add_argument("--loss", type=float, default=0.0, help="optional: drop this fraction of packets first")
    ap.add_argument("--jitter", type=float, default=0.0, help="optional: max extra delay per packet, seconds")
    ap.add_argument("--trunc", type=int, default=None, help="optional: observer sees only the first N packets")
    ap.add_argument("--csv", help="write per-flow predictions to this CSV")
    a = ap.parse_args()

    classes = load_classes(a.results, a.data)
    sizes, ts = D.pcap_to_flows(a.pcap, max_flows=a.max_flows)
    if len(sizes) == 0:
        raise SystemExit("no usable flows found in this pcap (need >= 5 packets per flow, non-DNS)")
    if a.loss or a.jitter or a.trunc:
        sizes, ts = perturb(sizes, ts, loss=a.loss, jitter=a.jitter, trunc=a.trunc, rng=np.random.default_rng(0))

    true = a.true_label or D.label_from_filename(a.pcap)
    if true not in classes:
        true = None
    print(f"\n{os.path.basename(a.pcap)}: {len(sizes)} flows"
          + (f" | true label: {true}" if true else " | true label: unknown")
          + (f" | stress: loss={a.loss} jitter={a.jitter}s trunc={a.trunc}" if (a.loss or a.jitter or a.trunc) else ""))

    models = find_models(a.results, a.model, a.seed)
    cols, w = [], max(len(c) for c in classes) + 1
    print(f"\n{'model':22s}" + "".join(f"{c[:w]:>{w + 1}s}" for c in classes) + "   verdict" + ("        flow acc" if true else ""))
    for variant, path in models.items():
        proba = predict_proba(variant, path, len(classes), sizes, ts)
        pred = proba.argmax(1)
        share = np.bincount(pred, minlength=len(classes)) / len(pred)
        line = f"{variant:22s}" + "".join(f"{100 * s:>{w}.1f}%" for s in share) + f"   {classes[share.argmax()]:12s}"
        if true:
            line += f"  {100 * (pred == classes.index(true)).mean():5.1f}%"
        print(line)
        cols.append((variant, pred, proba.max(1)))

    print("\n(columns = % of this pcap's flows assigned to each class; verdict = majority class)")
    if true is None:
        print("note: filename gave no label, so no accuracy is shown; use --true-label to add one")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["flow", "n_packets"] + [h for v, _, _ in cols for h in (f"{v}_pred", f"{v}_conf")])
            for i in range(len(sizes)):
                wr.writerow([i, int((sizes[i] != 0).sum())]
                            + [x for _, p, c in cols for x in (classes[p[i]], f"{c[i]:.3f}")])
        print(f"per-flow predictions written to {a.csv}")


if __name__ == "__main__":
    main()
