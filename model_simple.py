"""
Single-frame grayscale swipe detector (SimpleSwipeAnnotator).

Identical to model_simple.py in the training notebook; it lives here so
convert_to_coreml.py can rebuild the network from a train_simple.py checkpoint.

Input : x (B, 1, H, W) grayscale frame, raw 0..255 (the model divides by 255)
Output: det_logits (B,), dir_logits (B, 4) for [UP, DOWN, LEFT, RIGHT]
"""

from typing import Optional

import torch
import torch.nn as nn


class FrameCNN(nn.Module):
    """Single-channel conv feature extractor -> 128-d vector."""

    def __init__(self, in_channels: int = 1):
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

        self.b1 = block(in_channels, 32)
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


class SimpleSwipeAnnotator(nn.Module):
    def __init__(self, feat_dim: int = 128, head_hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.cnn = FrameCNN(in_channels=1)
        self.drop = nn.Dropout(dropout)
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.det_head = nn.Linear(head_hidden, 1)
        self.dir_head = nn.Linear(head_hidden, 4)

    def _cnn_features(self, x):
        x = x.to(dtype=torch.float32) * (1.0 / 255.0)
        x = x.to(memory_format=torch.channels_last)
        return self.cnn(x)

    def forward(self, x):
        feat = self.trunk(self.drop(self._cnn_features(x)))
        return self.det_head(feat).squeeze(-1), self.dir_head(feat)


def build_simple_model(arch: Optional[dict] = None) -> SimpleSwipeAnnotator:
    cfg = dict(feat_dim=128, head_hidden=128, dropout=0.3)
    if arch:
        cfg.update({k: arch[k] for k in cfg if k in arch})
    return SimpleSwipeAnnotator(**cfg)
