# Encrypted Traffic Type Classification (PyTorch + XGBoost)

Classifies network flows into traffic types (browsing, chat, email, file transfer, p2p, streaming, VoIP)
from the **first 32 packets only** (signed packet size, inter-arrival time, direction). No IP addresses,
ports or payload are used. It compares XGBoost, a 1D-CNN, an LSTM and a CNN + BiGRU + attention model,
each trained normally and with robustness augmentation, under controlled packet loss, jitter and truncation.

```
pcap files --> prepare_data.py --> flows.npz --> run.py --> results/ --> predict.py (test one pcap)
```

## 1. Files

| file | purpose |
|---|---|
| `prepare_data.py` | Step 1. pcap folder -> `flows.npz` (or synthetic data for a smoke test) |
| `run.py` | Step 2. trains all models x seeds, runs the network-condition tests, writes tables and plots |
| `predict.py` | Step 3. runs trained models on any single pcap |
| `data.py` | pcap parsing, filename -> label rules, dedup, train/val/test splits, synthetic data |
| `features.py` | per-packet tensors for the neural nets, flow statistics for XGBoost |
| `perturb.py` | packet loss, delay jitter, truncation (used for training augmentation and for testing) |
| `models.py` | `cnn1d`, `lstm`, `cnn_bigru`, `cnn_bigru_attn` |
| `requirements.txt` | Python dependencies |

## 2. Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows      (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

**GPU:** `pip install torch` on Windows installs the CPU-only build. For an NVIDIA GPU, get the install command
for your CUDA version from https://pytorch.org, then check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

`run.py` uses the GPU automatically when this prints `True`. Data preparation is CPU-only.

## 3. Dataset layout

Put all pcap folders under one parent (any depth, `.pcap` and `.pcapng` both work):

```
ISCXVPN2016/
  NonVPN-PCAPs-01/
  NonVPN-PCAPs-02/
  NonVPN-PCAPs-03/
```

Leave the `VPN-PCAPs-*` folders out. VPN traffic is tunnelled, so flow extraction would see a few huge
tunnel flows instead of individual applications.

Each pcap is labelled from its **filename** using `LABEL_RULES` in `data.py` (first match wins):

| class | filename contains |
|---|---|
| streaming | youtube, vimeo, netflix, spotify |
| p2p | torrent |
| chat | chat |
| email | email, mail |
| file_transfer | sftp, ftps, scp, file |
| voip | voip, audio, video |
| browsing | firefox, chrome, browsing |

Check the per-file lines printed in Step 1 and edit the rules if a capture lands in the wrong class. Classes with
fewer than 30 flows are dropped automatically when `run.py` loads the data.

## 4. Step 1: build the dataset

```bash
python prepare_data.py --pcap-dir ISCXVPN2016 --out flows.npz --workers 8 --max-flows-per-file 5000 --cache-dir cache
```

Prints one line per capture (flows found, assigned class, seconds) and the final per-class counts.
The one-time parse is the slow part; training afterwards only loads `flows.npz`.

| flag | default | meaning |
|---|---|---|
| `--pcap-dir DIR` | | folder to search recursively for `.pcap` / `.pcapng` |
| `--synthetic` | off | generate fake traffic instead (pipeline test only; never report its numbers) |
| `--out FILE` | `flows.npz` | output file |
| `--workers N` | 1 | parallel files; use about your physical core count |
| `--max-flows-per-file N` | none | randomly keep at most N flows per pcap. Shrinks the dataset and limits class imbalance; does **not** speed up parsing |
| `--max-packets-per-file N` | none | stop reading each pcap after N TCP/UDP packets. Faster, but only sees the start of each capture |
| `--cache-dir DIR` | none | save each parsed file so re-runs and added folders skip finished files |
| `--parser auto\|scapy` | `auto` | `auto` = fast dpkt parser, falling back to Scapy if needed |

Speed notes: a GPU does not help parsing. More workers than files or cores gains nothing, and the largest single
capture sets the minimum total time.

Pipeline check without any data:

```bash
python prepare_data.py --synthetic --out flows_synth.npz
```

## 5. Step 2: train and evaluate

```bash
python run.py --data flows.npz --out results --seeds 3
```

Quick pipeline check (3 epochs, 1 seed, ignore the numbers):

```bash
python run.py --data flows.npz --out results_test --quick
```

Other useful runs:

```bash
# stricter evaluation: whole capture files are held out of training
python run.py --data flows.npz --out results_group --split group

# only some models
python run.py --data flows.npz --out results_small --variants xgb xgb+aug cnn_bigru_attn cnn_bigru_attn+aug --seeds 1

# longer training
python run.py --data flows.npz --out results_long --epochs 80 --patience 15

# rebuild summary.md and robustness.png from the saved CSVs, no retraining
python run.py --out results --report-only
```

| flag | default | meaning |
|---|---|---|
| `--data FILE` | `flows.npz` | dataset from Step 1 |
| `--out DIR` | `results` | output folder |
| `--variants ...` | all 9 | space-separated list (see below) |
| `--seeds N` | 3 | independent training runs per variant |
| `--epochs N` | 40 | maximum epochs (early stopping applies) |
| `--batch-size N` | 256 | neural net batch size |
| `--lr X` | 0.002 | peak learning rate |
| `--patience N` | 8 | stop after N epochs without validation improvement |
| `--split flow\|group` | `flow` | `flow`: stratified random flows; `group`: whole pcap files held out |
| `--xgb-trials N` | 8 | random-search trials for XGBoost hyper-parameters |
| `--quick` | off | 3 epochs, 1 seed, 1 XGBoost trial |
| `--report-only` | off | rebuild summary and plot from existing CSVs |

**Variants.** `<model>` is trained on clean flows. `<model>+aug` is trained with random packet loss (0-12%),
jitter (0-120 ms) and truncation (5-32 packets). Default set: `xgb`, `xgb+aug`, `cnn1d`, `cnn1d+aug`, `lstm`,
`lstm+aug`, `cnn_bigru_attn`, `cnn_bigru_attn+aug` (main model), `cnn_bigru+aug` (no-attention ablation).

**Conditions tested** (every model sees identical perturbed test data):

| condition | levels | simulation |
|---|---|---|
| packet loss | 0, 1, 2, 5, 10, 20 % | each packet dropped independently |
| jitter | 0, 20, 50, 100, 200 ms | each packet delayed by U(0, J); large J reorders packets |
| truncation | 5, 10, 20, 32 packets | observer sees only the first N packets |

Loss 20% and jitter 200 ms are outside the training augmentation range, so they test extrapolation.

## 6. Step 3: test a single pcap

```bash
python predict.py --pcap test.pcap --results results --model cnn_bigru_attn+aug
python predict.py --pcap test.pcap --results results --model all
python predict.py --pcap test.pcap --results results --model all --loss 0.05 --jitter 0.05 --trunc 10
python predict.py --pcap test.pcap --results results --model all --csv per_flow.csv
```

| flag | default | meaning |
|---|---|---|
| `--pcap FILE` | required | capture to classify |
| `--results DIR` | `results` | folder produced by `run.py` |
| `--model NAME` | `all` | one variant name (e.g. `xgb`, `cnn1d+aug`) or `all` |
| `--seed N` | 0 | which training run's weights to use |
| `--true-label CLASS` | from filename | enables flow-level accuracy |
| `--loss X` | 0 | drop this fraction of packets before classifying |
| `--jitter S` | 0 | max extra delay per packet, in seconds |
| `--trunc N` | off | classifier sees only the first N packets |
| `--max-flows N` | 5000 | cap on flows read |
| `--csv FILE` | off | write per-flow predictions and confidences |
| `--data FILE` | | only needed if `classes.json` is missing |

Output: for each model, the % of the pcap's flows assigned to each class, the majority verdict, and flow-level
accuracy when the true label is known. Use a pcap that was **not** in the training folder.
A model can only answer with the classes it was trained on.

## 7. Output files (`results/`)

| file | contents |
|---|---|
| `summary.md` | main table: clean macro-F1 and accuracy (mean ± std over seeds), mean macro-F1 over all stressed conditions, parameters, latency |
| `robustness.csv` | every variant x seed x condition x level |
| `robustness.png` | degradation curves for loss, jitter, truncation |
| `confusion_matrix.png` | main model, clean test set |
| `efficiency.csv` | parameters (trees for XGBoost), single-flow latency, throughput |
| `classes.json` | class names, needed by `predict.py` |
| `ckpt/<variant>_seed<N>.pt` / `.json` | saved models (PyTorch weights / XGBoost) |

## 8. Troubleshooting

| problem | fix |
|---|---|
| `RuntimeError ... bootstrapping phase` on Windows | use the current `prepare_data.py`; it has the required `if __name__ == "__main__"` guard |
| `�` instead of `±` in `summary.md` | use the current `run.py`, then `python run.py --out results --report-only` |
| training is slow | check `torch.cuda.is_available()`; reduce `--seeds` or `--variants`; use Colab's free GPU (upload `flows.npz` and the `.py` files only) |
| a class is missing from the results | it had fewer than 30 flows and was dropped; add more captures or adjust `LABEL_RULES` |
| capture skipped with "no label rule" | add a keyword for it in `LABEL_RULES` |
| `predict.py`: no `classes.json` / no XGBoost model | the models came from an older `run.py`; re-run `run.py` |
| parsing is very slow | install `dpkt` (`pip install dpkt`), raise `--workers`, add `--max-packets-per-file` |
| a worker runs out of memory on a huge pcap | lower `--workers`, or use `--max-packets-per-file` |
| Scapy `WARNING` lines | harmless; they come from unusual packets in some captures |

## 9. Design choices and limitations

- Splits are by flow (stratified, fixed seed 42) after exact duplicates are removed. `--split group` is stricter
  because flows from one capture never appear in both train and test.
- Class-weighted loss (square-root inverse frequency) and macro-F1 as the headline metric, because classes are imbalanced.
- Model selection uses macro-F1 on the clean validation set for every variant.
- The ± in the results reflects training-seed variation only (one fixed split, one fixed test set).
- Labels come from capture filenames, so background flows inside a capture inherit that capture's label.
- Perturbations are simulated on packet arrays: reproducible, but not real network behaviour. Replaying traffic through
  Linux `tc netem` or Mininet and capturing it again is the more realistic upgrade.
- ISCXVPN2016 is TLS 1.2-era traffic; modern TLS 1.3, QUIC and Encrypted Client Hello traffic may behave differently.
