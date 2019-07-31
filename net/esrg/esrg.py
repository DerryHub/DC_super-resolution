import torch
from torch import nn

import os
import sys

root = os.path.dirname(__file__)
sys.path.append(root)

from esrg_utils import ResidualInResidualDenseBlock, Shuffle


class ESRG(nn.Module):
    def __init__(self, n=4, beta=0.2, num=16):
        super(ESRG, self).__init__()
        self.firstConv = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.blocks = [
            ResidualInResidualDenseBlock(in_channels=64, beta=beta)
            for _ in range(num)
        ]
        self.blocks.append(nn.Conv2d(64, 64, 3, 1, padding=1))
        self.BBs = nn.Sequential(*self.blocks)
        self.shuffle = Shuffle(64, n=n)
        self.lastConv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.Conv2d(64, 3, 3, 1, padding=1))

    def forward(self, x):
        o1 = self.firstConv(x)
        o2 = o1 + self.BBs(o1)
        output = self.shuffle(o2)
        output = self.lastConv(output)
        return output
