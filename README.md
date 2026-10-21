# Encrypted Traffic Type Classification (PyTorch + XGBoost)

Classifies flows into traffic types (browsing, chat, email, file transfer, p2p, streaming, VoIP) from the
**first 32 packets** only (signed size, inter-arrival time, direction). No IPs, ports, or payload are used.
Compares XGBoost / 1D-CNN / LSTM against a CNN + BiGRU + attention model, with and without robustness
training, under controlled packet loss, jitter, and truncation.

## Files

| file | purpose |
|---|---|
| `prepare_data.py` | pcap folder -> `flows.npz` (or `--synthetic` for a smoke test) |
| `data.py` | flow extraction (Scapy), filename -> label rules, dedup, flow/group splits |
| `features.py` | sequence tensors for the nets, flow statistics for XGBoost (same 32-packet window) |
| `perturb.py` | packet loss, delay jitter (+ natural reordering), truncation; used for training and testing |
| `models.py` | `cnn1d`, `lstm`, `cnn_bigru` (no attention), `cnn_bigru_attn` (main) |
| `run.py` | trains all variants x seeds, runs the condition sweeps, writes tables and plots |

## Quickstart

```bash
pip install -r requirements.txt

# 1. build the dataset (ISCXVPN2016 pcaps, any folder depth)
python prepare_data.py --pcap-dir /path/to/ISCXVPN2016 --out flows.npz --workers 4 --max-flows-per-file 5000

# 2. check the printed per-class counts, then run everything (3 seeds)
python run.py --data flows.npz --out results --seeds 3

# stricter evaluation: hold out whole capture files instead of random flows
python run.py --data flows.npz --out results_group --split group
```

Pipeline check without any dataset: `python prepare_data.py --synthetic --out s.npz && python run.py --data s.npz --quick`.
The synthetic traffic only exists to test the code. **Never report numbers from it.**

## Before you trust the labels

`LABEL_RULES` in `data.py` maps pcap filenames to the 7 classes by keyword (e.g. `*_chat*` -> chat,
`youtube/vimeo/netflix/spotify` -> streaming, `skype_audio` -> voip). I could not check it against your
actual download, so look at the per-file lines printed by `prepare_data.py` and edit the rules if a capture
lands in the wrong class or is skipped. Classes with fewer than 30 flows are dropped automatically.

## Variants and what they show

`<model>` = trained on clean flows, `<model>+aug` = trained with random loss (0-12%), jitter (0-120 ms) and
truncation (5-32 packets). Main model: `cnn_bigru_attn+aug`.

- `xgb+aug`, `cnn1d+aug`, `lstm+aug` are controls: they show whether any gain comes from the augmentation
  itself or from the architecture.
- `cnn_bigru+aug` (no attention) vs `cnn_bigru_attn+aug` is the attention ablation.
- XGBoost gets a small random hyper-parameter search (`--xgb-trials`, selected on validation macro-F1).

## Test conditions

| condition | levels | how it is simulated |
|---|---|---|
| packet loss | 0, 1, 2, 5, 10, **20** % | each packet dropped independently |
| jitter | 0, 20, 50, 100, **200** ms | each packet delayed by U(0, J); large J reorders packets |
| truncation | 5, 10, 20, 32 packets | observer only sees the first N packets |

Bold levels are outside the training augmentation range, so they test extrapolation. Each condition is applied
once to the test set with a fixed seed, so every model sees identical data.

## Outputs (`results/`)

- `summary.md`: clean macro-F1 / accuracy (mean ± std over seeds), mean macro-F1 over all stressed conditions, parameters, latency
- `robustness.csv`: every variant x seed x condition
- `robustness.png`: degradation curves; `confusion_matrix.png`: main model on clean test data
- `efficiency.csv`: parameters, single-flow latency (CPU, batch 1), throughput (flows/s)
- `ckpt/`: saved PyTorch weights

## Design choices worth stating in your report

- Splits are by flow (stratified, fixed seed 42) with exact-duplicate flows removed first; `--split group` is stricter.
- Class-weighted loss (sqrt inverse frequency) and macro-F1 as the headline metric, because classes are imbalanced.
- Model selection uses macro-F1 on the clean validation set for every variant.
- Perturbations are simulated on packet arrays, which is reproducible but not real network behaviour. The optional
  upgrade is to replay traffic through Linux `tc netem` / Mininet and test on the captures.
- ISCXVPN2016 is TLS 1.2-era traffic; modern TLS 1.3 / QUIC / ECH traffic may behave differently.
