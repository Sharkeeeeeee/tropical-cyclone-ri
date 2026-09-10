"""Small CNN for RI classification from IR patch sequences.

About a quarter-width ResNet-18 with GroupNorm and a 64-dimensional embedding.
Sized for ~1,100 independent RI episodes in the whole archive; anything with
tens of millions of parameters would memorise storms.

Frames enter as channels rather than through a 3D convolution -- with four
frames that is the cheaper way to let the first layer see differences between
them.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = gn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = gn(cout)
        self.act = nn.ReLU(inplace=True)
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                   gn(cout)))

    def forward(self, x):
        y = self.act(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return self.act(y + self.skip(x))


class PatchCNN(nn.Module):
    def __init__(self, in_frames: int = 4, width: int = 16, emb_dim: int = 64,
                 p_drop: float = 0.3):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(in_frames, w, 5, 2, 2, bias=False), gn(w),
            nn.ReLU(inplace=True))                        # 128 -> 64
        self.stages = nn.Sequential(
            ResBlock(w, w),
            ResBlock(w, 2 * w, stride=2),                 # 64 -> 32
            ResBlock(2 * w, 4 * w, stride=2),             # 32 -> 16
            ResBlock(4 * w, 4 * w),
            ResBlock(4 * w, 8 * w, stride=2),             # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Sequential(nn.Flatten(), nn.Linear(8 * w, emb_dim),
                                   nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(emb_dim, 1))

    def features(self, x):
        return self.embed(self.pool(self.stages(self.stem(x))))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
