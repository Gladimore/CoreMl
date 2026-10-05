"""
Causal dual-rate (fast/slow) swipe-detection architecture.

Design
------
- Fast branch : 8 frames, native rate, strictly [t-7 ... t]        (fine motion)
- Slow branch : 2 frames, at offsets (t-19, t-13), i.e. strictly older than the
                fast window                                         (long-range context)
- One shared per-frame CNN (2 channels: frame + diff -> 128-d feature)
- Each branch is encoded by its own unidirectional (causal) GRU that starts
  from a ZERO hidden state for every window
- The last hidden state of each GRU is concatenated and fed to the det / dir heads

Training / deployment contract
------------------------------
`forward()` is a *windowed* model: every sample is evaluated from a zero hidden
state over exactly 8 fast + 2 slow frames. Deployment must reproduce exactly
that computation, which is what `convert_to_coreml.py` exports:

    SwipeEncoder : (frame, diff) -> 128-d feature            (run once per new frame)
    SwipeHead    : 8 fast feats + 2 slow feats -> det/dir    (run once per tick)

`heads_from_features()` below is the reference implementation of the second
stage and is what the exported head is verified against.

The streaming `init_state()` / `step()` API is kept only for backwards
compatibility. It is NOT equivalent to `forward()` (it carries hidden state
indefinitely, which the model never saw during training) and must not be used
for deployment.

Input contract
--------------
forward(): fast_x (B, 8, 2, H, W), slow_x (B, 2, 2, H, W), values 0..255

Output
------
det_logits : (B,)     current-frame detection logit
dir_logits : (B, 4)   direction logits (UP, DOWN, LEFT, RIGHT)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

FAST_LEN = 8
SLOW_OFFSETS = (19, 13)          # frames behind t, oldest first
FEAT_DIM = 128


# ─────────────────────────────────────────────────────────────────────────────
# Shared per-frame backbone
# ─────────────────────────────────────────────────────────────────────────────

class FrameCNN(nn.Module):
    """Per-frame 2-channel (frame, diff) conv feature extractor -> 128-d vector."""

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
        # x: (N, 2, H, W)
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.pool(self.b3(x))
        x = self.gap(self.b4(x))
        return x.flatten(1)          # (N, 128)


# ─────────────────────────────────────────────────────────────────────────────
# Causal dual-branch temporal model
# ─────────────────────────────────────────────────────────────────────────────

class CausalSwipeAnnotator(nn.Module):
    def __init__(self,
                 hidden: int = 192,
                 fast_layers: int = 2,
                 slow_layers: int = 1,
                 dropout: float = 0.3):
        super().__init__()

        self.hidden = hidden
        self.fast_layers = fast_layers
        self.slow_layers = slow_layers
        self.slow_hidden = hidden // 2

        self.cnn = FrameCNN()

        self.fast_gru = nn.GRU(
            input_size=FEAT_DIM,
            hidden_size=hidden,
            num_layers=fast_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if fast_layers > 1 else 0.0,
        )
        self.slow_gru = nn.GRU(
            input_size=FEAT_DIM,
            hidden_size=self.slow_hidden,
            num_layers=slow_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if slow_layers > 1 else 0.0,
        )

        fused_dim = hidden + self.slow_hidden
        self.drop = nn.Dropout(dropout)
        self.det_head = nn.Linear(fused_dim, 1)
        self.dir_head = nn.Linear(fused_dim, 4)

    # ── stage 1: per-frame features ─────────────────────────────────────────

    def _cnn_features(self, x):
        """x: (N, 2, H, W) raw 0..255 (uint8 or float) -> (N, 128) features."""
        x = x.to(dtype=torch.float32) * (1.0 / 255.0)
        x = x.to(memory_format=torch.channels_last)
        return self.cnn(x)

    def encode_sequence(self, x):
        """x: (B, T, 2, H, W) -> (B, T, 128)."""
        B, T, C, H, W = x.shape
        feats = self._cnn_features(x.reshape(B * T, C, H, W))
        return feats.view(B, T, -1)

    # ── stage 2: temporal heads over cached features ────────────────────────

    def heads_from_features(self, fast_feats, slow_feats):
        """
        fast_feats : (B, 8, 128)  features of frames [t-7 ... t]
        slow_feats : (B, 2, 128)  features of frames [t-19, t-13]
        Both GRUs start from a zero hidden state.
        """
        fast_out, _ = self.fast_gru(fast_feats)
        slow_out, _ = self.slow_gru(slow_feats)
        fused = torch.cat([fast_out[:, -1, :], slow_out[:, -1, :]], dim=-1)
        fused = self.drop(fused)
        return self.det_head(fused).squeeze(-1), self.dir_head(fused)

    # ── windowed forward (training) ─────────────────────────────────────────

    def forward(self, fast_x, slow_x):
        return self.heads_from_features(self.encode_sequence(fast_x),
                                        self.encode_sequence(slow_x))

    # ── legacy streaming API (NOT train/deploy equivalent — see docstring) ──

    def init_state(self, batch_size: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or next(self.parameters()).device
        h_fast = torch.zeros(self.fast_layers, batch_size, self.hidden, device=device)
        h_slow = torch.zeros(self.slow_layers, batch_size, self.slow_hidden, device=device)
        return h_fast, h_slow

    def step(self, fast_frame, h_fast, h_slow, slow_frame: Optional[torch.Tensor] = None):
        fast_feat = self._cnn_features(fast_frame).unsqueeze(1)
        _, h_fast_new = self.fast_gru(fast_feat, h_fast)
        if slow_frame is not None:
            slow_feat = self._cnn_features(slow_frame).unsqueeze(1)
            _, h_slow_new = self.slow_gru(slow_feat, h_slow)
        else:
            h_slow_new = h_slow
        fused = self.drop(torch.cat([h_fast_new[-1], h_slow_new[-1]], dim=-1))
        return (self.det_head(fused).squeeze(-1), self.dir_head(fused),
                h_fast_new, h_slow_new)


def build_causal_model(arch: Optional[dict] = None) -> CausalSwipeAnnotator:
    """Build model from an arch dict (e.g. from a checkpoint) or defaults."""
    cfg = dict(hidden=192, fast_layers=2, slow_layers=1, dropout=0.3)
    if arch:
        cfg.update({k: arch[k] for k in cfg if k in arch})
    return CausalSwipeAnnotator(**cfg)
