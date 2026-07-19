"""
Causal dual-rate (fast/slow) swipe-detection architecture — v2.

Design summary
---------------
- Fast branch : 8 frames, native rate, strictly [t-7 ... t]        (fine motion, ~291ms @ 24fps)
- Slow branch : 2 frames, ~4fps-equivalent stride, strictly < t-7  (long-range context, ~500-800ms back)
- Both branches share one CNN backbone (per-frame 2ch frame+diff features)
- Each branch is encoded by its own unidirectional (causal) GRU
- Final hidden states from both GRUs are concatenated and fed to det/dir heads
- Model is causal end-to-end: at inference time frame t's output depends only on
  frames <= t. No frame from the future is ever consumed.

CHANGES vs v1 of this file
---------------------------
Added a second inference mode: `init_state()` + `step()`, for O(1)-per-frame
streaming deployment. The original `forward(fast_x, slow_x)` windowed mode
recomputed the CNN over all 10 frames on every call — correct, but wasteful
if called once per tick at 24fps (effectively a 240fps CNN workload for a
24fps stream). `step()` instead consumes exactly one new frame and carries
GRU hidden state forward between calls, so the CNN runs once per new frame.

IMPORTANT — train/deploy correspondence caveat (read before using `step`):
`forward()` (training) always starts each window from a zero hidden state,
i.e. it approximates the fast branch's receptive field as *exactly* the
last 8 frames, discarding everything before that. `step()` (deployment), by
contrast, carries hidden state indefinitely from the start of the session —
it does not reset every 8 frames. These are NOT mathematically identical:
`step()`'s live hidden state will retain some (heavily decayed) influence
from frames further back than 8, which the model never explicitly saw
during training. This is a standard, widely-used approximation for
deploying windowed-trained RNNs in a streaming fashion (comparable to how
truncated-BPTT-trained RNNs are commonly deployed with continuous state),
and is expected to work well here given GRU forget dynamics naturally decay
old information — but it is an approximation, not a proof of equivalence,
and should be validated empirically (e.g. compare `step()`-produced logits
against `forward()`-produced logits on held-out sessions) before trusting
it in production.

Input contract
--------------
forward(): fast_x (B, 8, 2, H, W), slow_x (B, 2, 2, H, W) — windowed, training use
step():    fast_frame (B, 2, H, W) every tick; slow_frame (B, 2, H, W) or None
           on ticks with no new slow-branch sample — streaming, deployment use

Output
------
det_logits : (B,)     single current-frame detection logit (target = frame t only)
dir_logits : (B, 4)   direction logits (UP, DOWN, LEFT, RIGHT), current-frame only
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Shared per-frame backbone (unchanged — no reason to redesign it)
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

        self.b1   = block(2,   32)
        self.b2   = block(32,  64)
        self.b3   = block(64,  128)
        self.b4   = block(128, 128)
        self.pool = nn.MaxPool2d(2)
        self.gap  = nn.AdaptiveAvgPool2d(1)

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
    """
    Causal, dual-rate (fast/slow) online swipe detector.

    Both branches share one FrameCNN backbone (weight-tied feature extraction),
    then are each encoded by their own unidirectional GRU. Final hidden states
    are concatenated and passed to the detection/direction heads.

    No component of this model has access to any frame later than the current
    frame t — the fast GRU is unidirectional and the slow branch is, by
    construction, sampled strictly before the fast branch's earliest frame.
    """

    def __init__(self,
                 hidden: int = 192,
                 fast_layers: int = 2,
                 slow_layers: int = 1,
                 dropout: float = 0.3):
        super().__init__()

        self.hidden      = hidden
        self.fast_layers = fast_layers
        self.slow_layers = slow_layers
        self.slow_hidden = hidden // 2

        # Single shared backbone — both branches produce 128-d per-frame features
        self.cnn = FrameCNN()

        # Fast branch: dense recent motion, causal GRU (unidirectional)
        self.fast_gru = nn.GRU(
            input_size    = 128,
            hidden_size   = hidden,
            num_layers    = fast_layers,
            batch_first   = True,
            bidirectional = False,
            dropout       = dropout if fast_layers > 1 else 0.0,
        )

        # Slow branch: sparse long-range context, causal GRU (unidirectional)
        self.slow_gru = nn.GRU(
            input_size    = 128,
            hidden_size   = self.slow_hidden,
            num_layers    = slow_layers,
            batch_first   = True,
            bidirectional = False,
            dropout       = dropout if slow_layers > 1 else 0.0,
        )

        fused_dim = hidden + self.slow_hidden

        self.drop     = nn.Dropout(dropout)
        self.det_head = nn.Linear(fused_dim, 1)
        self.dir_head = nn.Linear(fused_dim, 4)

    # ── shared feature extraction ───────────────────────────────────────────

    def _cnn_features(self, x):
        """x: (N, 2, H, W) uint8 or float -> (N, 128) float features."""
        x = x.to(dtype=torch.float32) * (1.0 / 255.0)
        x = x.to(memory_format=torch.channels_last)
        return self.cnn(x)

    # ── windowed mode (training) ────────────────────────────────────────────

    def _encode_branch_window(self, x, gru):
        """
        x   : (B, T, 2, H, W)
        gru : the branch's causal GRU
        returns final hidden state (B, hidden) — depends only on x[:, 0..T-1]
        in order, i.e. purely on frames <= the branch's own last frame.
        Hidden state starts at zero for every call (see module docstring for
        the train/deploy correspondence caveat this implies).
        """
        B, T, C, H, W = x.shape
        x_2d  = x.reshape(B * T, C, H, W)
        feats = self._cnn_features(x_2d)
        feats = feats.view(B, T, -1)               # (B, T, 128)
        gru_out, _ = gru(feats)                     # (B, T, hidden)
        return gru_out[:, -1, :]                     # (B, hidden)

    def forward(self, fast_x, slow_x):
        """
        Windowed mode — used for training via BPTT over sampled windows.

        fast_x : (B, 8, 2, H, W)  frames [t-7 ... t], native rate, most recent last
        slow_x : (B, 2, 2, H, W)  frames strictly older than fast_x's oldest frame,
                                   sparse stride, most recent (of the slow set) last

        Returns:
            det_logits : (B,)    current-frame (t) detection logit
            dir_logits : (B, 4)  current-frame (t) direction logits
        """
        fast_feat = self._encode_branch_window(fast_x, self.fast_gru)   # (B, hidden)
        slow_feat = self._encode_branch_window(slow_x, self.slow_gru)   # (B, slow_hidden)

        fused = torch.cat([fast_feat, slow_feat], dim=-1)        # (B, fused_dim)
        fused = self.drop(fused)

        det_logits = self.det_head(fused).squeeze(-1)            # (B,)
        dir_logits = self.dir_head(fused)                        # (B, 4)
        return det_logits, dir_logits

    # ── streaming mode (deployment) ─────────────────────────────────────────

    def init_state(self, batch_size: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Zero-initialized hidden states for the start of a streaming session.
        Call once when a new session/game begins; thread the returned state
        through subsequent step() calls.

        Returns (h_fast, h_slow):
            h_fast : (fast_layers, B, hidden)
            h_slow : (slow_layers, B, slow_hidden)
        """
        device = device or next(self.parameters()).device
        h_fast = torch.zeros(self.fast_layers, batch_size, self.hidden, device=device)
        h_slow = torch.zeros(self.slow_layers, batch_size, self.slow_hidden, device=device)
        return h_fast, h_slow

    def step(self,
              fast_frame: torch.Tensor,
              h_fast: torch.Tensor,
              h_slow: torch.Tensor,
              slow_frame: Optional[torch.Tensor] = None):
        """
        Single-tick streaming update. CNN runs once for fast_frame, and once
        more for slow_frame only on ticks where a new slow-branch sample
        exists (e.g. every 6th tick, matching the ~4fps slow-branch stride
        used during dataset construction). On all other ticks, pass
        slow_frame=None and h_slow is carried through unchanged.

        fast_frame : (B, 2, H, W)              the new current frame's (frame, diff) pair
        slow_frame : (B, 2, H, W) or None       new slow-branch sample, if this tick has one
        h_fast     : (fast_layers, B, hidden)   carried from the previous call
        h_slow     : (slow_layers, B, slow_hidden)  carried from the previous call

        Returns:
            det_logits : (B,)
            dir_logits : (B, 4)
            h_fast_new : (fast_layers, B, hidden)   pass into the next step() call
            h_slow_new : (slow_layers, B, slow_hidden)  pass into the next step() call
        """
        fast_feat = self._cnn_features(fast_frame).unsqueeze(1)      # (B, 1, 128)
        _, h_fast_new = self.fast_gru(fast_feat, h_fast)              # h_fast_new: (fast_layers, B, hidden)

        if slow_frame is not None:
            slow_feat = self._cnn_features(slow_frame).unsqueeze(1)   # (B, 1, 128)
            _, h_slow_new = self.slow_gru(slow_feat, h_slow)           # (slow_layers, B, slow_hidden)
        else:
            h_slow_new = h_slow                                        # unchanged this tick

        fused = torch.cat([h_fast_new[-1], h_slow_new[-1]], dim=-1)   # (B, fused_dim)
        fused = self.drop(fused)

        det_logits = self.det_head(fused).squeeze(-1)                 # (B,)
        dir_logits = self.dir_head(fused)                             # (B, 4)
        return det_logits, dir_logits, h_fast_new, h_slow_new


def build_causal_model(arch: Optional[dict] = None) -> CausalSwipeAnnotator:
    """Build model from an arch dict (e.g. from checkpoint) or defaults."""
    defaults = dict(hidden=192, fast_layers=2, slow_layers=1, dropout=0.3)
    if arch:
        defaults.update({k: arch[k] for k in
                          ('hidden', 'fast_layers', 'slow_layers', 'dropout')
                          if k in arch})
    return CausalSwipeAnnotator(**defaults)
