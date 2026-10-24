"""Train every model variant, evaluate under controlled network conditions, write tables + plots.

    python run.py --data flows.npz --out results --seeds 3
    python run.py --data flows_synth.npz --out results_smoke --quick        # fast pipeline check

Variants:  <model>  = trained on clean flows      <model>+aug = trained with robustness augmentation
Models:    xgb (flow statistics) | cnn1d | lstm | cnn_bigru_attn (main model) | cnn_bigru (no-attention ablation)
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import data as D
from features import flow_stats, seq_features
from models import build, n_params
from perturb import perturb, sample_conditions

DEFAULT_VARIANTS = ["xgb", "xgb+aug", "cnn1d", "cnn1d+aug", "lstm", "lstm+aug",
                    "cnn_bigru_attn", "cnn_bigru_attn+aug", "cnn_bigru+aug"]
MAIN = "cnn_bigru_attn+aug"

# Controlled conditions. loss = per-packet drop probability, jitter = max extra delay per packet (s),
# trunc = number of packets the observer sees. Loss 20% and jitter 200 ms lie OUTSIDE the augmentation
# range (<=12% / 120 ms), so they test extrapolation rather than memorised conditions.
CONDITIONS = {
    "loss":   [0.0, 0.01, 0.02, 0.05, 0.10, 0.20],
    "jitter": [0.0, 0.02, 0.05, 0.10, 0.20],
    "trunc":  [5, 10, 20, 32],
}


# --------------------------------------------------------------------------- helpers
def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    return (len(y) / (n_classes * counts)) ** 0.5          # sqrt-inverse-frequency: gentle re-balancing


def build_test_conditions(S, T, seed=12345):
    """Perturb the test set ONCE per condition with a fixed seed -> every model sees identical data."""
    sets = {}
    for kind, levels in CONDITIONS.items():
        for lv in levels:
            s, t = perturb(S, T, rng=np.random.default_rng(seed), **{kind: lv})
            sets[(kind, lv)] = (seq_features(s, t), flow_stats(s, t))
    return sets


@torch.no_grad()
def predict_nn(model, X, device, bs=2048):
    model.eval()
    return torch.cat([model(torch.from_numpy(X[i:i + bs]).to(device)).argmax(1).cpu()
                      for i in range(0, len(X), bs)]).numpy()


# --------------------------------------------------------------------------- neural nets
def train_nn(base, aug, seed, tr, va, S, T, Y, n_classes, args, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build(base, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * int(np.ceil(len(tr) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    cw = torch.tensor(class_weights(Y[tr], n_classes), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    Xva, yva = seq_features(S[va], T[va]), Y[va]

    best, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            if len(idx) < 2:
                continue
            s, t = S[idx], T[idx]
            if aug:
                s, t = perturb(s, t, rng=rng, **sample_conditions(len(idx), rng))
            x = torch.from_numpy(seq_features(s, t)).to(device)
            y = torch.from_numpy(Y[idx]).to(device)
            loss = crit(model(x), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        f1 = f1_score(yva, predict_nn(model, Xva, device), average="macro")
        if f1 > best + 1e-4:
            best, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


# --------------------------------------------------------------------------- XGBoost
def _xgb(params, seed, early=True):
    from xgboost import XGBClassifier
    return XGBClassifier(**params, random_state=seed, tree_method="hist", n_jobs=-1,
                         eval_metric="mlogloss", early_stopping_rounds=30 if early else None)


def tune_xgb(S, T, Y, tr, va, n_classes, trials, seed=0):
    """Small random search on clean data, selected by validation macro-F1 (so the baseline is not a strawman)."""
    rng = np.random.default_rng(seed)
    Xtr, Xva = flow_stats(S[tr], T[tr]), flow_stats(S[va], T[va])
    w = class_weights(Y[tr], n_classes)[Y[tr]]
    cands = [dict(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=1, reg_lambda=1.0)]
    for _ in range(trials - 1):
        cands.append(dict(n_estimators=int(rng.choice([200, 400, 800])), max_depth=int(rng.integers(3, 10)),
                          learning_rate=float(rng.choice([0.03, 0.05, 0.1, 0.2])),
                          subsample=float(rng.uniform(0.6, 1.0)), colsample_bytree=float(rng.uniform(0.5, 1.0)),
                          min_child_weight=int(rng.integers(1, 8)), reg_lambda=float(rng.choice([0.5, 1, 5, 10]))))
    best, best_p = -1, cands[0]
    for p in cands:
        clf = _xgb(p, seed).fit(Xtr, Y[tr], sample_weight=w, eval_set=[(Xva, Y[va])], verbose=False)
        f1 = f1_score(Y[va], clf.predict(Xva), average="macro")
        if f1 > best:
            best, best_p = f1, p
    print(f"  xgb tuning: best val macro-F1 {best:.4f} with {best_p}")
    return best_p


def train_xgb(aug, seed, params, tr, va, S, T, Y, n_classes):
    rng = np.random.default_rng(seed)
    Xs, ys = [flow_stats(S[tr], T[tr])], [Y[tr]]
    if aug:                                              # clean copy + 2 randomly perturbed copies
        for _ in range(2):
            s, t = perturb(S[tr], T[tr], rng=rng, **sample_conditions(len(tr), rng, p_clean=0.0))
            Xs.append(flow_stats(s, t)); ys.append(Y[tr])
    Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
    w = class_weights(Y[tr], n_classes)[ytr]
    clf = _xgb(params, seed).fit(Xtr, ytr, sample_weight=w,
                                 eval_set=[(flow_stats(S[va], T[va]), Y[va])], verbose=False)
    return clf


# --------------------------------------------------------------------------- efficiency
def time_nn(model, X):
    m = copy.deepcopy(model).cpu().eval()
    x1, xb = torch.from_numpy(X[:1]), torch.from_numpy(X[:1024])
    with torch.no_grad():
        for _ in range(20):
            m(x1)
        t0 = time.perf_counter()
        for _ in range(200):
            m(x1)
        lat = (time.perf_counter() - t0) / 200 * 1000
        t0 = time.perf_counter()
        for _ in range(5):
            m(xb)
        thr = 5 * len(xb) / (time.perf_counter() - t0)
    return lat, thr


def time_xgb(clf, X):
    x1, xb = X[:1], X[:1024]
    clf.predict(x1)
    t0 = time.perf_counter()
    for _ in range(100):
        clf.predict(x1)
    lat = (time.perf_counter() - t0) / 100 * 1000
    t0 = time.perf_counter()
    clf.predict(xb)
    return lat, len(xb) / (time.perf_counter() - t0)


# --------------------------------------------------------------------------- reporting
def make_plots(rob, cm, classes, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"xgb": "tab:gray", "cnn1d": "tab:blue", "lstm": "tab:green",
              "cnn_bigru_attn": "tab:red", "cnn_bigru": "tab:orange"}
    titles = {"loss": "Packet loss", "jitter": "Delay jitter (max extra delay)", "trunc": "Packets observed"}
    xlabels = {"loss": "loss rate (%)", "jitter": "jitter (ms)", "trunc": "first N packets"}
    scale = {"loss": 100, "jitter": 1000, "trunc": 1}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, kind in zip(axes, CONDITIONS):
        for v, g in rob[rob.kind == kind].groupby("variant"):
            base, _, tag = v.partition("+")
            agg = g.groupby("level").macro_f1.agg(["mean", "std"]).fillna(0)
            x = agg.index.values * scale[kind]
            ax.plot(x, agg["mean"], "-o" if tag else "--s", color=colors.get(base, "k"), label=v,
                    lw=2.2 if v == MAIN else 1.4, ms=4)
            ax.fill_between(x, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            color=colors.get(base, "k"), alpha=0.12)
        ax.set_title(titles[kind]); ax.set_xlabel(xlabels[kind]); ax.set_ylabel("macro-F1"); ax.grid(alpha=.3)
        ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("Macro-F1 under controlled network conditions (solid = robustness-trained, dashed = clean-trained)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "robustness.png"), dpi=160); plt.close(fig)

    if cm is not None:
        name, y_true, y_pred = cm
        mat = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true")
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] > .5 else "black", fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Confusion matrix (clean test) - {name}")
        fig.tight_layout(); fig.savefig(os.path.join(out, "confusion_matrix.png"), dpi=160); plt.close(fig)


def write_summary(rob, eff, out):
    clean = rob[(rob.kind == "loss") & (rob.level == 0.0)]
    stressed = rob[~((rob.kind == "loss") & (rob.level == 0.0)) & ~((rob.kind == "jitter") & (rob.level == 0.0))
                   & ~((rob.kind == "trunc") & (rob.level == 32))]
    lines = ["| model | clean macro-F1 | clean accuracy | mean macro-F1 over all stressed conditions | params | latency (ms/flow) |",
             "|---|---|---|---|---|---|"]
    for v in rob.variant.unique():
        c, s = clean[clean.variant == v], stressed[stressed.variant == v]
        e = eff[eff.variant == v].iloc[0]
        lines.append(f"| {v} | {c.macro_f1.mean():.4f} ± {c.macro_f1.std(ddof=0):.4f} | "
                     f"{c.accuracy.mean():.4f} ± {c.accuracy.std(ddof=0):.4f} | "
                     f"{s.macro_f1.mean():.4f} ± {s.groupby('seed').macro_f1.mean().std(ddof=0):.4f} | "
                     f"{e.params}{' trees' if v.startswith('xgb') else ''} | {e.latency_ms:.2f} |")
    text = "\n".join(lines)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    try:
        print("\n" + text)
    except UnicodeEncodeError:                      # exotic console code pages
        print("\n" + text.replace("±", "+/-"))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="flows.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--split", choices=["flow", "group"], default="flow",
                    help="flow: stratified by flow | group: whole pcap files held out (stricter)")
    ap.add_argument("--xgb-trials", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="3 epochs, 1 seed, no XGB tuning: pipeline check only")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild summary.md and robustness.png from the CSVs already in --out (no training)")
    args = ap.parse_args()
    if args.report_only:
        rob = pd.read_csv(os.path.join(args.out, "robustness.csv"))
        eff = pd.read_csv(os.path.join(args.out, "efficiency.csv"))
        write_summary(rob, eff, args.out)
        make_plots(rob, None, [], args.out)         # confusion_matrix.png is left untouched
        print(f"\nrebuilt summary.md and robustness.png in {args.out}/")
        return
    if args.quick:
        args.epochs, args.seeds, args.xgb_trials = 3, 1, 1

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    S, T, Y, G, classes = D.load(args.data)
    n_classes = len(classes)
    json.dump(classes, open(os.path.join(args.out, "classes.json"), "w"))   # needed by predict.py
    tr, va, te = D.make_splits(Y, G, args.split)
    print(f"device={device} | {len(Y)} flows, {n_classes} classes | split={args.split}: "
          f"train {len(tr)} / val {len(va)} / test {len(te)}")
    for i, c in enumerate(classes):
        print(f"  {c:14s} train {int((Y[tr] == i).sum()):6d}  test {int((Y[te] == i).sum()):6d}")

    cond = build_test_conditions(S[te], T[te])
    y_te = Y[te]
    xgb_params = tune_xgb(S, T, Y, tr, va, n_classes, args.xgb_trials) if any(v.startswith("xgb") for v in args.variants) else None

    rob_rows, eff_rows, cm = [], [], None
    for variant in args.variants:
        base, _, tag = variant.partition("+")
        aug = tag == "aug"
        for seed in range(args.seeds):
            t0 = time.time()
            if base == "xgb":
                model = train_xgb(aug, seed, xgb_params, tr, va, S, T, Y, n_classes)
                predict = lambda sets: model.predict(sets[1])
                info = f"trees={model.best_iteration + 1}"
                model.save_model(os.path.join(args.out, "ckpt", f"{variant}_seed{seed}.json"))
            else:
                model, val_f1, ep = train_nn(base, aug, seed, tr, va, S, T, Y, n_classes, args, device)
                predict = lambda sets: predict_nn(model, sets[0], device)
                info = f"epochs={ep} val_f1={val_f1:.4f}"
                torch.save(model.state_dict(), os.path.join(args.out, "ckpt", f"{variant}_seed{seed}.pt"))
            for (kind, lv), sets in cond.items():
                pred = predict(sets)
                rob_rows.append(dict(variant=variant, seed=seed, kind=kind, level=lv,
                                     macro_f1=f1_score(y_te, pred, average="macro"),
                                     accuracy=accuracy_score(y_te, pred)))
                if kind == "loss" and lv == 0.0 and seed == 0 and (variant == MAIN or cm is None):
                    cm = (variant, y_te, pred)
            clean = [r for r in rob_rows[-len(cond):] if r["kind"] == "loss" and r["level"] == 0.0][0]
            print(f"[{variant:20s} seed {seed}] clean macro-F1 {clean['macro_f1']:.4f} "
                  f"acc {clean['accuracy']:.4f} | {info} | {time.time() - t0:.0f}s")
            if seed == 0:
                lat, thr = time_xgb(model, cond[("loss", 0.0)][1]) if base == "xgb" else time_nn(model, cond[("loss", 0.0)][0])
                eff_rows.append(dict(variant=variant, params=(model.best_iteration + 1) if base == "xgb" else n_params(model),
                                     latency_ms=lat, throughput_flows_s=thr))
            pd.DataFrame(rob_rows).to_csv(os.path.join(args.out, "robustness.csv"), index=False)   # incremental save

    rob, eff = pd.DataFrame(rob_rows), pd.DataFrame(eff_rows)
    eff.to_csv(os.path.join(args.out, "efficiency.csv"), index=False)
    write_summary(rob, eff, args.out)
    make_plots(rob, cm, classes, args.out)
    print(f"\nAll outputs written to {args.out}/")


if __name__ == "__main__":
    main()
