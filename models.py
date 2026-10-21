"""PyTorch models. Input everywhere: x of shape (B, 3, N) from features.seq_features.
Channel 2 is the packet direction, which is 0 exactly on padding -> used as the validity mask."""
import torch
import torch.nn as nn


def _mask(x):
    m = x[:, 2, :] != 0
    m[:, 0] = True            # never an empty sequence (e.g. every packet dropped)
    return m


class CNN1D(nn.Module):
    """Baseline 1: plain 1D-CNN over the packet sequence."""
    def __init__(self, n_classes, in_ch=3, width=64, dropout=0.3):
        super().__init__()
        def block(i, o, k): return [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU()]
        self.features = nn.Sequential(*block(in_ch, width, 5), *block(width, width * 2, 5), nn.MaxPool1d(2),
                                      *block(width * 2, width * 2, 3))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 4, n_classes))

    def forward(self, x):
        h = self.features(x)
        return self.head(torch.cat([h.mean(2), h.amax(2)], dim=1))


class LSTMNet(nn.Module):
    """Baseline 2: 2-layer LSTM, masked mean pooling over the valid packets."""
    def __init__(self, n_classes, in_ch=3, hidden=96, dropout=0.3):
        super().__init__()
        self.rnn = nn.LSTM(in_ch, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        m = _mask(x).unsqueeze(-1).float()
        h, _ = self.rnn(x.transpose(1, 2))
        return self.head((h * m).sum(1) / m.sum(1))


class CNNBiGRU(nn.Module):
    """Main model: conv stem (local packet patterns) -> BiGRU (flow dynamics) -> attention pooling.
    use_attn=False swaps attention for masked mean pooling (ablation)."""
    def __init__(self, n_classes, in_ch=3, width=64, hidden=64, dropout=0.3, use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        def block(i, o): return [nn.Conv1d(i, o, 3, padding=1), nn.BatchNorm1d(o), nn.ReLU()]
        self.stem = nn.Sequential(*block(in_ch, width), *block(width, width))
        self.gru = nn.GRU(width, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden if use_attn else 2 * hidden, n_classes))

    def forward(self, x, return_attn=False):
        m = _mask(x)
        h, _ = self.gru(self.stem(x).transpose(1, 2))                  # (B, N, 2H)
        mf = m.unsqueeze(-1).float()
        mean = (h * mf).sum(1) / mf.sum(1)
        if not self.use_attn:
            return self.head(mean)
        score = self.attn(h).squeeze(-1).masked_fill(~m, float("-inf"))
        w = torch.softmax(score, dim=1)                                # (B, N) attention over packets
        ctx = (w.unsqueeze(-1) * h).sum(1)
        out = self.head(torch.cat([ctx, mean], dim=1))
        return (out, w) if return_attn else out


def build(name, n_classes):
    if name == "cnn1d":
        return CNN1D(n_classes)
    if name == "lstm":
        return LSTMNet(n_classes)
    if name == "cnn_bigru_attn":
        return CNNBiGRU(n_classes, use_attn=True)
    if name == "cnn_bigru":
        return CNNBiGRU(n_classes, use_attn=False)
    raise ValueError(name)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
