"""
TemporalSwipeNet — early-fusion multi-frame swipe detector.

Why this design (vs. the old causal GRU model and the simple model)
-------------------------------------------------------------------
Speed and "what is moving where" are PIXEL-LEVEL facts: an obstacle 3 px closer
than last frame, the track scrolling 2 px vs 5 px per frame. The old causal model
squeezed every frame to a 128-d global-average-pooled vector before the GRU ever
compared frames, so most of that displacement information was already gone, and
it paid for it with 10 CNN passes per training sample.

This model stacks a few past frames as input CHANNELS ("early fusion"), so the
very first convolution compares the same pixel across time and can measure
motion and speed directly:

    input  (B, K, H, W)   K grayscale frames at tick offsets `offsets`
                          (default 0,1,2,4,8: now, 1, 2, 4 and 8 ticks ago);
                          x[:, 0] is the newest frame. Raw 0..255.
    output det_logits (B,), dir_logits (B, 4)  [UP, DOWN, LEFT, RIGHT]

The dyadic offsets give motion baselines of 1, 2, 4 and 8 ticks, which covers
both slow and fast running speeds, and ~1/3 s of context (mid-jump, just
swiped, ...) at 24 fps, for almost no extra compute: only the first conv sees K
channels.

Other choices, all for speed and accuracy:
  - stride-2 stem: the old net ran two 32-channel convs at full 128x128, which was
    ~40% of its compute for edge detection. Here the first conv downsamples.
    ~176M MACs per prediction vs ~690M for the simple model (4x cheaper to train
    and run), ~0.85M params.
  - spatial head instead of global average pooling: a 1x1 conv squeezes the
    final 8x8 map to 32 channels, which is flattened, so the head knows WHERE
    things are (which lane the obstacle is in, how far down the track).
    GAP throws that away.
  - no recurrence and no hidden state: one Core ML model, one call per tick.
    The device keeps a ring of the last max(offsets)+1 grayscale frames.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn

DEFAULT_OFFSETS = (0, 1, 2, 4, 8)


def _conv_bn(cin: int, cout: int, stride: int = 1):
    return [nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class TemporalSwipeNet(nn.Module):
    def __init__(self, offsets: Sequence[int] = DEFAULT_OFFSETS, img_size: int = 128,
                 width: int = 32, head_ch: int = 32, head_hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        if img_size % 16:
            raise ValueError("img_size must be a multiple of 16")
        self.offsets = tuple(int(o) for o in offsets)
        if self.offsets[0] != 0 or len(set(self.offsets)) != len(self.offsets) or min(self.offsets) < 0:
            raise ValueError("offsets must start with 0 (the current frame) and be unique, >= 0")
        k, w = len(self.offsets), width

        self.features = nn.Sequential(
            *_conv_bn(k, w, stride=2), *_conv_bn(w, w),                 # S/2
            nn.MaxPool2d(2), *_conv_bn(w, 2 * w), *_conv_bn(2 * w, 2 * w),          # S/4
            nn.MaxPool2d(2), *_conv_bn(2 * w, 4 * w), *_conv_bn(4 * w, 4 * w),      # S/8
            nn.MaxPool2d(2), *_conv_bn(4 * w, 4 * w), *_conv_bn(4 * w, 4 * w),      # S/16
        )
        self.squeeze = nn.Sequential(nn.Conv2d(4 * w, head_ch, 1, bias=False),
                                     nn.BatchNorm2d(head_ch), nn.ReLU(inplace=True))
        side = img_size // 16
        self.trunk = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                   nn.Linear(head_ch * side * side, head_hidden),
                                   nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.det_head = nn.Linear(head_hidden, 1)
        self.dir_head = nn.Linear(head_hidden, 4)

    def forward(self, x):
        """x: (B, K, H, W) raw 0..255 (uint8 or float); x[:, k] is the frame offsets[k] ticks ago."""
        x = x.to(torch.float32) * (1.0 / 255.0)
        h = self.trunk(self.squeeze(self.features(x)))
        return self.det_head(h).squeeze(-1), self.dir_head(h)


ARCH_KEYS = ("offsets", "img_size", "width", "head_ch", "head_hidden", "dropout")


def build_temporal_model(arch: Optional[dict] = None) -> TemporalSwipeNet:
    cfg = dict(offsets=DEFAULT_OFFSETS, img_size=128, width=32, head_ch=32, head_hidden=128, dropout=0.3)
    if arch:
        cfg.update({k: arch[k] for k in ARCH_KEYS if k in arch})
    return TemporalSwipeNet(**cfg)
