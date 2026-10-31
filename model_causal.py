"""
Causal dual-rate (fast/slow) swipe detector (CausalSwipeAnnotator).

- Fast branch: 8 frames [t-7 .. t]
- Slow branch: 2 frames at t-19 and t-13 (oldest first)
- One shared per-frame CNN over (gray, diff) -> 128-d feature
- Each branch is a unidirectional GRU started from a ZERO hidden state for
  every window; the two last hidden states feed the det / dir heads.

Deployment reproduces forward() exactly (see convert_to_coreml.py):
    SwipeEncoder : (gray, diff) -> 128-d feature      once per tick, cached on device
    SwipeHead    : 8 fast + 2 slow cached features    once per tick, zero-state GRUs

Input : fast_x (B, 8, 2, H, W), slow_x (B, 2, 2, H, W), raw 0..255
Output: det_logits (B,), dir_logits (B, 4) for [UP, DOWN, LEFT, RIGHT]
"""

from typing import Optional

import torch
import torch.nn as nn

FAST_LEN = 8
SLOW_OFFSETS = (19, 13)          # frames behind t, oldest first
FEAT_DIM = 128


class FrameCNN(nn.Module):
    """Per-frame 2-channel (gray, diff) conv feature extractor -> 128-d vector."""

    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.b1 = block(2, 32)
        self.b2 = block(32, 64)
        self.b3 = block(64, 128)
        self.b4 = block(128, 128)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.pool(self.b3(x))
        x = self.gap(self.b4(x))
        return x.flatten(1)


class CausalSwipeAnnotator(nn.Module):
    def __init__(self, hidden: int = 192, fast_layers: int = 2, slow_layers: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.hidden = hidden
        self.fast_layers = fast_layers
        self.slow_layers = slow_layers
        self.slow_hidden = hidden // 2

        self.cnn = FrameCNN()
        self.fast_gru = nn.GRU(FEAT_DIM, hidden, num_layers=fast_layers, batch_first=True,
                               dropout=dropout if fast_layers > 1 else 0.0)
        self.slow_gru = nn.GRU(FEAT_DIM, self.slow_hidden, num_layers=slow_layers, batch_first=True,
                               dropout=dropout if slow_layers > 1 else 0.0)

        fused_dim = hidden + self.slow_hidden
        self.drop = nn.Dropout(dropout)
        self.det_head = nn.Linear(fused_dim, 1)
        self.dir_head = nn.Linear(fused_dim, 4)

    def _cnn_features(self, x):
        """x: (N, 2, H, W) raw 0..255 -> (N, 128)."""
        x = x.to(dtype=torch.float32) * (1.0 / 255.0)
        x = x.to(memory_format=torch.channels_last)
        return self.cnn(x)

    def encode_sequence(self, x):
        """x: (B, T, 2, H, W) -> (B, T, 128)."""
        B, T, C, H, W = x.shape
        return self._cnn_features(x.reshape(B * T, C, H, W)).view(B, T, -1)

    def heads_from_features(self, fast_feats, slow_feats):
        """fast_feats (B, 8, 128) for [t-7 .. t]; slow_feats (B, 2, 128) for [t-19, t-13]."""
        fast_out, _ = self.fast_gru(fast_feats)
        slow_out, _ = self.slow_gru(slow_feats)
        fused = self.drop(torch.cat([fast_out[:, -1, :], slow_out[:, -1, :]], dim=-1))
        return self.det_head(fused).squeeze(-1), self.dir_head(fused)

    def forward(self, fast_x, slow_x):
        return self.heads_from_features(self.encode_sequence(fast_x),
                                        self.encode_sequence(slow_x))


def build_causal_model(arch: Optional[dict] = None) -> CausalSwipeAnnotator:
    cfg = dict(hidden=192, fast_layers=2, slow_layers=1, dropout=0.3)
    if arch:
        cfg.update({k: arch[k] for k in cfg if k in arch})
    return CausalSwipeAnnotator(**cfg)
